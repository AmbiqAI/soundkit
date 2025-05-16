import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Warm-up + Cosine Decay Scheduler
class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self,
            initial_lr,
            total_steps,
            warmup_steps,
            alpha=0.0,
            initial_step=500):
        super().__init__()
        
        self.initial_step=initial_step
        self.initial_lr = initial_lr
        self.decay_steps = total_steps - warmup_steps
        self.warmup_steps = warmup_steps
        self.alpha = alpha
        self.cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=self.decay_steps,
            alpha=alpha
        )

    def __call__(self, step):
        step = tf.cast(step+self.initial_step, tf.float32)
        warmup_lr = self.initial_lr * step / tf.cast(self.warmup_steps, tf.float32)
        return tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: self.cosine_decay(step - self.warmup_steps)
        )

# Parameters
initial_lr = 1e-3
initial_step = 500
total_steps = 13000
warmup_steps = 1000


alpha = 1e-5


# Instantiate schedule

lr_schedule = WarmUpCosineDecay(initial_lr, total_steps, warmup_steps, alpha, initial_step=500)

# Collect LR values
steps = np.arange(total_steps - initial_step)
lrs = [lr_schedule(step).numpy() for step in steps]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(steps+500, lrs)
plt.title("Warm-up + Cosine Decay Learning Rate Schedule")
plt.xlabel("Training Step")
plt.ylabel("Learning Rate")
plt.grid(True)
plt.tight_layout()
plt.show()
