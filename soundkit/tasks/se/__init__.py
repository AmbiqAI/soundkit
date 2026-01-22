"""soundKIT SE Task Module"""
from soundkit.tasks.task import SKTask
from soundkit.defines import SKTaskParams
from .data import data
from .train import train
from .evaluate import evaluate
from .export import export
from .demo import demo

class SeTask(SKTask):
    """soundKIT SE Task"""

    @staticmethod
    def description() -> str:
        return (
            "This task is used to train, evaluate, and export se models."
        )

    @staticmethod
    def data(params: SKTaskParams):
        """ Data preparation for se task

        Args:
            params (SKTaskParams): Task parameters
        """
        data(params)

    @staticmethod
    def train(params: SKTaskParams):
        """Train model for se task

        Args:
            params (SKTaskParams): Task parameters
        """
        train(params)

    @staticmethod
    def evaluate(params: SKTaskParams):
        """Evaluate beat se model

        Args:
            params (SKTaskParams): Task parameters
        """
        evaluate(params)

    @staticmethod
    def export(params: SKTaskParams):
        """Export model for se task

        Args:
            params (SKTaskParams): Task parameters
        """
        export(params)

    @staticmethod
    def demo(params: SKTaskParams):
        """Run demo on se task model

        Args:
            params (SKTaskParams): Task parameters
        """
        demo(params)
