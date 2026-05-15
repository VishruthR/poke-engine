use crate::engine::evaluate::evaluate;
use crate::engine::generate_instructions::generate_instructions_from_move_pair;
use crate::engine::state::MoveChoice;
use crate::instruction::StateInstructions;
use crate::state::State;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rng;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::time::Duration;

const GAMMA: f64 = 0.1;
const CURR_STRAT_FLOOR: f64 = 1e-6;

fn sigmoid(x: f32) -> f32 {
    // Tuned so that ~200 points is very close to 1.0
    1.0 / (1.0 + (-0.0125 * x).exp())
}

#[derive(Debug)]
pub struct Node {
    pub root: bool,
    pub parent: *mut Node,
    pub children: HashMap<(usize, usize), Vec<Node>>,
    pub times_visited: u32,
    pub total_score: f32,
    pub depth: u16,

    // represents the instructions & s1/s2 moves that led to this node from the parent
    pub instructions: StateInstructions,
    pub s1_choice: u8,
    pub s2_choice: u8,

    // represents the total score and number of visits for this node
    // de-coupled for s1 and s2
    pub s1_options: Option<Vec<MoveNode>>,
    pub s2_options: Option<Vec<MoveNode>>,
}

impl Node {
    fn new() -> Node {
        Node {
            root: false,
            parent: std::ptr::null_mut(),
            instructions: StateInstructions::default(),
            times_visited: 0,
            total_score: 0.0,
            depth: 0,
            children: HashMap::new(),
            s1_choice: 0,
            s2_choice: 0,
            s1_options: None,
            s2_options: None,
        }
    }
    unsafe fn populate(&mut self, s1_options: Vec<MoveChoice>, s2_options: Vec<MoveChoice>) {
        let s1_options_vec: Vec<MoveNode> = s1_options
            .iter()
            .map(|x| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
                cum_strat: 0.0,
                curr_strat: 0.0,
                est_score: 0.0,
            })
            .collect();
        let s2_options_vec: Vec<MoveNode> = s2_options
            .iter()
            .map(|x| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
                cum_strat: 0.0,
                curr_strat: 0.0,
                est_score: 0.0,
            })
            .collect();

        self.s1_options = Some(s1_options_vec);
        self.s2_options = Some(s2_options_vec);
    }

    pub fn exp3_strategy(side_map: &[MoveNode], gamma: f64) -> Vec<f64> {
        let k = side_map.len() as f64;
        let lr = gamma / k;
        let exploration = gamma / k;
        let max_score = side_map
            .iter()
            .map(|m| m.est_score)
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_weights: Vec<f64> = side_map
            .iter()
            .map(|m| (lr * (m.est_score - max_score)).exp())
            .collect();
        let denom: f64 = exp_weights.iter().sum();
        exp_weights
            .iter()
            .map(|w| (1.0 - gamma) * (w / denom) + exploration)
            .collect()
    }

    pub fn exp3_selection(side_map: &mut [MoveNode]) -> usize {
        let strategy = Self::exp3_strategy(side_map, GAMMA);

        // Update cumulative strategies
        for (idx, move_strat) in strategy.iter().enumerate() {
            side_map[idx].curr_strat = *move_strat;
            side_map[idx].cum_strat += *move_strat;
        }

        let mut rng = rng();
        WeightedIndex::new(&strategy).unwrap().sample(&mut rng)
    }

    /*
    s1 and s2 choose their best move based on ucb1
    Checks if the turn (s1, s2) has been played

    If so, it applies the instructions and then recurses
    if not, it will return for another step to enumerate possibilities

    TLDR: Walk down existing tree down best path until you hit an unexplored node and return
    */
    pub unsafe fn selection(&mut self, state: &mut State) -> (*mut Node, usize, usize) {
        let return_node = self as *mut Node;
        if self.s1_options.is_none() {
            let (s1_options, s2_options) = state.get_all_options();
            self.populate(s1_options, s2_options);
        }

        let s1_mc_index = Self::exp3_selection(self.s1_options.as_mut().unwrap());
        let s2_mc_index = Self::exp3_selection(self.s2_options.as_mut().unwrap());

        // Bucket of stochastic outcomes for this (s1, s2) pair (e.g. 30% burn).
        let child_vector = self.children.get_mut(&(s1_mc_index, s2_mc_index));
        match child_vector {
            Some(child_vector) => {
                let child_vec_ptr = child_vector as *mut Vec<Node>;
                let chosen_child = self.sample_node(child_vec_ptr);
                state.apply_instructions(&(*chosen_child).instructions.instruction_list);
                (*chosen_child).selection(state)
            }
            None => (return_node, s1_mc_index, s2_mc_index),
        }
    }

    unsafe fn sample_node(&self, move_vector: *mut Vec<Node>) -> *mut Node {
        let mut rng = rng();
        let weights: Vec<f64> = (*move_vector)
            .iter()
            .map(|x| x.instructions.percentage as f64)
            .collect();
        let dist = WeightedIndex::new(weights).unwrap();
        let chosen_node = &mut (&mut *move_vector)[dist.sample(&mut rng)];
        let chosen_node_ptr = chosen_node as *mut Node;
        chosen_node_ptr
    }

    /*
     * Generate all possibilites based on (s1, s2) move choice
     * Store under the (s1, s2) node in a move_vector
     * Sample an outcome weighted by probability and elaborate/roll-out on it
     *
     * only branches on damage rolls on the first two levels (more than that would be too hard)
     */
    pub unsafe fn expand(
        &mut self,
        state: &mut State,
        s1_move_index: usize,
        s2_move_index: usize,
    ) -> (*mut Node, u32) {
        let s1_move = &self.s1_options.as_ref().unwrap()[s1_move_index].move_choice;
        let s2_move = &self.s2_options.as_ref().unwrap()[s2_move_index].move_choice;
        // if the battle is over or both moves are none there is no need to expand
        if (state.battle_is_over() != 0.0 && !self.root)
            || (s1_move == &MoveChoice::None && s2_move == &MoveChoice::None)
        {
            return (self as *mut Node, 0);
        }
        debug_assert!(self.root || !self.parent.is_null());
        let should_branch_on_damage = self.root || (*self.parent).root;
        let mut new_instructions =
            generate_instructions_from_move_pair(state, s1_move, s2_move, should_branch_on_damage);
        let mut this_pair_vec = Vec::with_capacity(new_instructions.len());
        let child_depth = self.depth.saturating_add(1);
        for state_instructions in new_instructions.drain(..) {
            let mut new_node = Node::new();
            new_node.parent = self;
            new_node.instructions = state_instructions;
            new_node.s1_choice = s1_move_index as u8;
            new_node.s2_choice = s2_move_index as u8;
            new_node.depth = child_depth;

            this_pair_vec.push(new_node);
        }

        let nodes_added = this_pair_vec.len() as u32;
        // sample a node from the new instruction list.
        // this is the node that the rollout will be done on
        let new_node_ptr = self.sample_node(&mut this_pair_vec);
        state.apply_instructions(&(*new_node_ptr).instructions.instruction_list);
        self.children
            .insert((s1_move_index, s2_move_index), this_pair_vec);
        (new_node_ptr, nodes_added)
    }

    /*
     * backpropagate this nodes result to the root and intermediate nodes
     * This way root gets rewarded for good outcomes
     */
    pub unsafe fn backpropagate(&mut self, score: f32, state: &mut State) {
        self.times_visited += 1;
        if self.root {
            return;
        }

        let parent_s1_movenode =
            &mut (*self.parent).s1_options.as_mut().unwrap()[self.s1_choice as usize];
        parent_s1_movenode.total_score += score;
        parent_s1_movenode.visits += 1;
        parent_s1_movenode.est_score +=
            score as f64 / parent_s1_movenode.curr_strat.max(CURR_STRAT_FLOOR);
        assert!(
            parent_s1_movenode.est_score.is_finite(),
            "exp3 s1 est_score is not finite"
        );

        let parent_s2_movenode =
            &mut (*self.parent).s2_options.as_mut().unwrap()[self.s2_choice as usize];
        parent_s2_movenode.total_score += 1.0 - score;
        parent_s2_movenode.visits += 1;
        parent_s2_movenode.est_score +=
            (1.0 - score) as f64 / parent_s2_movenode.curr_strat.max(CURR_STRAT_FLOOR);
        assert!(
            parent_s2_movenode.est_score.is_finite(),
            "exp3 s2 est_score is not finite"
        );

        state.reverse_instructions(&self.instructions.instruction_list);
        (*self.parent).backpropagate(score, state);
        
    }

    /*
     * Return a score based on the current battle state 
     */
    pub fn rollout(&mut self, state: &mut State, root_eval: &f32) -> f32 {
        let battle_is_over = state.battle_is_over();
        if battle_is_over == 0.0 {
            let eval = evaluate(state);
            sigmoid(eval - root_eval)
        } else {
            if battle_is_over == -1.0 {
                0.0
            } else {
                battle_is_over
            }
        }
    }
}

#[derive(Debug)]
pub struct MoveNode {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: u32,
    pub cum_strat: f64,
    pub curr_strat: f64,
    pub est_score: f64,
}

impl MoveNode {
    /*
    calculates upper confidence bound. Goal is to balance exploitation vs. exploration
    first term (total score / visits) is exploitation
        you calculate the moves average score per visit 
    second term (parent visits term)
        tries to visit previously unvisited moves since it grows inversely to a moves visits
        higher c = more exploration

    unvisited moves get INF, this is to ensure a move gets visited at least once before any move gets visited twice in a given level

    at the end, the move with the most visits wins (not average score) since ucb pushes towards moves with high confidence (high average could be due to luck)

    TLDR: Prefer moves that have looked good in the past but give a boost to moves we haven't tested much
    */
    pub fn ucb1(&self, parent_visits: u32) -> f32 {
        if self.visits == 0 {
            return f32::INFINITY;
        }
        let score = (self.total_score / self.visits as f32)
            + (2.0 * (parent_visits as f32).ln() / self.visits as f32).sqrt();
        score
    }
    pub fn average_score(&self) -> f32 {
        let score = self.total_score / self.visits as f32;
        score
    }
}

pub fn average_strategy(side_map: &[MoveNode], gamma: f64) -> Vec<f32> {
    let k = side_map.len();
    let uniform = 1.0 / k as f32;
    // cum_strat is incremented for every arm on every selection traversal of this
    // node, so the correct denominator (eq. 12) is the total number of traversals,
    // which equals the sum of per-arm visits at this node.
    let total_visits: u64 = side_map.iter().map(|m| m.visits as u64).sum();
    if total_visits == 0 {
        return vec![uniform; k];
    }
    let t = total_visits as f64;
    let floor = gamma / k as f64;
    let unnormalized: Vec<f64> = side_map
        .iter()
        .map(|m| ((m.cum_strat / t) - floor).max(0.0))
        .collect();
    let sum: f64 = unnormalized.iter().sum();
    if sum <= 0.0 {
        return vec![uniform; k];
    }
    unnormalized.iter().map(|s| (s / sum) as f32).collect()
}

#[derive(Clone)]
pub struct MctsSideResult {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: u32,
    pub avg_strat: f32,
}

impl MctsSideResult {
    pub fn average_score(&self) -> f32 {
        if self.visits == 0 {
            return 0.0;
        }
        let score = self.total_score / self.visits as f32;
        score
    }
}

pub struct MctsResult {
    pub s1: Vec<MctsSideResult>,
    pub s2: Vec<MctsSideResult>,
    pub iteration_count: u32,
}

/*
 * Traverses current tree for the best unexplored move
 * Enumerates its possibilities and samples one
 * Calculates score of this possibility relative to root
 * Backpropagates score up path
 */
fn do_mcts(root_node: &mut Node, state: &mut State, root_eval: &f32) -> (u16, u32) {
    let (selected, s1_move, s2_move) = unsafe { root_node.selection(state) };
    let (new_node, nodes_added) = unsafe { (*selected).expand(state, s1_move, s2_move) };
    let rollout_result = unsafe { (*new_node).rollout(state, root_eval) };
    let leaf_depth = unsafe { (*new_node).depth };
    unsafe { (*new_node).backpropagate(rollout_result, state) };
    (leaf_depth, nodes_added)
}

pub fn perform_mcts_exp3(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
) -> MctsResult {
    let mut root_node = Node::new();
    unsafe {
        root_node.populate(side_one_options, side_two_options);
    }
    root_node.root = true;

    let root_eval = evaluate(state);
    let start_time = std::time::Instant::now();
    let mut nodes_expanded: u64 = 0;
    let mut max_depth: u16 = 0;
    let mut total_depth: u64 = 0;
    while start_time.elapsed() < max_time {
        for _ in 0..1000 {
            let (leaf_depth, added) = do_mcts(&mut root_node, state, &root_eval);
            nodes_expanded += added as u64;
            total_depth += leaf_depth as u64;
            if leaf_depth > max_depth {
                max_depth = leaf_depth;
            }
        }

        /*
        Cut off after 10 million iterations

        Under normal circumstances the bot will only run for 2.5-3.5 million iterations
        however towards the end of a battle the bot may perform tens of millions of iterations

        Beyond about 30 million iterations some floating point nonsense happens where
        MoveNode.total_score stops updating because f32 does not have enough precision

        I can push the problem farther out by using f64 but if the bot is running for 10 million iterations
        then it almost certainly sees a forced win
        */
        if root_node.times_visited >= 10_000_000 {
            break;
        }
    }

    let elapsed_ms = start_time.elapsed().as_millis();
    let iterations = root_node.times_visited;
    let avg_depth = if iterations > 0 {
        total_depth as f64 / iterations as f64
    } else {
        0.0
    };
    let log_path = std::env::var("POKE_MCTS_STATS_LOG")
        .unwrap_or_else(|_| format!("/Users/vishruthraj/Code/CS498AlgoEng/final_project/poke-engine/logs/poke_mcts_stats_exp3_{}.log", std::process::id()));
    if let Ok(file) = OpenOptions::new().create(true).append(true).open(&log_path) {
        let mut w = BufWriter::new(file);
        let _ = writeln!(
            w,
            "elapsed_ms={} iterations={} nodes_expanded={} max_depth={} avg_depth={:.2}",
            elapsed_ms, iterations, nodes_expanded, max_depth, avg_depth
        );
    }

    let s1_avg = average_strategy(root_node.s1_options.as_ref().unwrap(), GAMMA);
    let s2_avg = average_strategy(root_node.s2_options.as_ref().unwrap(), GAMMA);

    let result = MctsResult {
        s1: root_node
            .s1_options
            .as_ref()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, v)| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
                avg_strat: s1_avg[i],
            })
            .collect(),
        s2: root_node
            .s2_options
            .as_ref()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, v)| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
                avg_strat: s2_avg[i],
            })
            .collect(),
        iteration_count: root_node.times_visited,
    };

    result
}
