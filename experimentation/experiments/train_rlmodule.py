from absl import app, flags

from experimentation.experiments import experiment_rlmodule

FLAGS = flags.FLAGS
flags.DEFINE_string("experiment_name", None, "Name of the experiment")
flags.DEFINE_boolean("debug", False, "Debug mode flag")


def main(*args, **kwargs):
    experiment_rlmodule.Experiment(
        config={
            "debug": FLAGS.debug,
            "experiment_name": FLAGS.experiment_name,
        }
    ).run()


if __name__ == "__main__":
    app.run(main)
