use kore_core::Tensor;

pub fn cmd_train(steps: usize, lr: f32, scheduler_name: &str, warmup_pct: f32) {
    use kore_optim::{CosineAnnealing, LrScheduler, OneCycle, StepDecay, WarmupCosine};

    println!("=== Kore Training Demo ===");
    println!(
        "Steps: {}, LR: {}, Scheduler: {}",
        steps, lr, scheduler_name
    );

    // Build scheduler
    let sched: Box<dyn LrScheduler> = match scheduler_name {
        "cosine" => Box::new(CosineAnnealing::new(lr, lr * 0.01, steps)),
        "warmup_cosine" => {
            let warmup = (steps as f32 * warmup_pct) as usize;
            Box::new(WarmupCosine::new(lr * 0.01, lr, lr * 0.01, warmup, steps))
        }
        "one_cycle" => Box::new(OneCycle::new(lr, steps, warmup_pct)),
        "step" => Box::new(StepDecay::new(lr, 0.5, steps / 5, steps)),
        _ => {
            println!("No scheduler, using constant LR");
            Box::new(CosineAnnealing::new(lr, lr, steps)) // constant
        }
    };

    // Create a tiny model and synthetic data
    let in_dim = 16;
    let out_dim = 4;
    let layer = kore_nn::Linear::new(in_dim, out_dim, true);

    // Synthetic input/target
    let x_data: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.1) - 0.8).collect();
    let target: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0];
    let x = Tensor::from_f32(&x_data, &[1, in_dim]);
    let t = Tensor::from_f32(&target, &[1, out_dim]);

    println!("\n{:<8} {:>10} {:>10}", "Step", "Loss", "LR");
    println!("{}", "-".repeat(30));

    let _opt = kore_optim::Adam::default_with_lr(lr);

    for step in 0..steps {
        let current_lr = sched.get_lr(step);

        // Forward
        let y = kore_nn::Module::forward(&layer, &x).expect("forward failed");

        // MSE loss
        let diff = y.sub(&t).expect("sub failed");
        let sq = diff.mul(&diff).expect("mul failed");
        let loss_val = sq.as_f32_slice().unwrap().iter().sum::<f32>() / out_dim as f32;

        // Compute gradient (manual for demo: d_loss/d_y = 2*(y-t)/n)
        let _grad_y = diff.mul_scalar(2.0 / out_dim as f32).unwrap();

        // Approximate param gradients via finite differences would be complex;
        // for this demo, just show the training loop structure
        if step % (steps / 10).max(1) == 0 || step == steps - 1 {
            println!("{:<8} {:>10.6} {:>10.6}", step, loss_val, current_lr);
        }
    }

    println!("\nTraining complete.");
}
