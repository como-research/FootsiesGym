from absl import app, flags

from experimentation.experiments.rllib import experiment

FLAGS = flags.FLAGS
flags.DEFINE_string("experiment_name", None, "Name of the experiment")
flags.DEFINE_boolean("debug", False, "Debug mode flag")
flags.DEFINE_boolean("tune", False, "Tune mode flag")


def main(*args, **kwargs):
    print(f"Starting experiment {FLAGS.experiment_name}, Tuning: {FLAGS.tune}")
    experiment.Experiment(
        config={
            "debug": FLAGS.debug,
            "experiment_name": FLAGS.experiment_name,
            "tune": FLAGS.tune,
            "checkpoint_freq": 10,
        }
    ).run()


if __name__ == "__main__":
    app.run(main)
