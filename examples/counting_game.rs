extern crate mcts;

use mcts::*;

#[derive(Clone, Copy)]
struct CountingGame(i32);

impl GameState for CountingGame {
    type Move = CountingGame;
    type GameResult = ();

    fn result(&self) -> Option<Self::GameResult> {
        None
    }

    fn available_moves(&self) -> Vec<Self::Move> {
        let x = self.0;
        vec![CountingGame(x - 1), CountingGame(x + 1)]
    }

    fn make_move(&mut self, mov: &Self::Move) {
        *self = *mov;
    }
}

use std::cell::*;

struct MyEvaluator {}

impl Evaluator<MyMCTS> for MyEvaluator {
    type StateEvaluation = i64;
    type MoveEvaluation = i64;

    fn evaluate_state_and_moves(&self, state: &CountingGame, moves: &[CountingGame],
        handle: SearchHandle<MyMCTS>)
        -> (i64, Vec<i64>) {
        (0, moves.iter().map(|_| 0).collect())
    }

    fn interpret_evaluation_for_current_player(&self, state: &CountingGame, evaln: &i64) -> i64 {
        *evaln
    }
}

struct MyMCTS {}

impl MCTS for MyMCTS {
    type State = CountingGame;
    type Eval = MyEvaluator;
    type NodeData = ();
    type ThreadLocalData = ();
    type GlobalData = ();

}

fn main() {

}
