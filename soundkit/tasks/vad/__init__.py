from ..task import SKTask
from ...defines import SKTaskParams
from .data import data
from .train import train
from .evaluate import evaluate
from .export import export
from .demo import demo

class VadTask(SKTask):
    """HeartKit Beat Task"""

    @staticmethod
    def description() -> str:
        return (
            "This task is used to train, evaluate, and export beat models."
            "Beat includes normal, pac, pvc, and other beats."
        )

    @staticmethod
    def data(params: SKTaskParams):
        """ Data preparation for se task

        Args:
            params (HKTaskParams): Task parameters
        """
        data(params)

    @staticmethod
    def train(params: SKTaskParams):
        """Train model for se task

        Args:
            params (HKTaskParams): Task parameters
        """
        train(params)

    @staticmethod
    def evaluate(params: SKTaskParams):
        """Evaluate beat se model

        Args:
            params (HKTaskParams): Task parameters
        """
        evaluate(params)

    @staticmethod
    def export(params: SKTaskParams):
        """Export model for se task

        Args:
            params (HKTaskParams): Task parameters
        """
        export(params)

    @staticmethod
    def demo(params: SKTaskParams):
        """Run demo on se task model

        Args:
            params (HKTaskParams): Task parameters
        """
        demo(params)
