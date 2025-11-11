"""SoundKit Task base class module."""
import abc # abstract base class

from ..defines import SKTaskParams


class SKTask(abc.ABC):
    """SoundKit Task base class. All tasks should inherit from this class."""

    @staticmethod
    def description() -> str:
        """Get task description

        Returns:
            str: description

        """
        return ""

    @staticmethod
    def data(params: SKTaskParams) -> None:
        """Data preparation for the task

        Args:
            params (SKTaskParams): Task parameters

        """
        raise NotImplementedError
    
    @staticmethod
    def train(params: SKTaskParams) -> None:
        """Train a model

        Args:
            params (SKTaskParams): Task parameters

        """
        raise NotImplementedError

    @staticmethod
    def evaluate(params: SKTaskParams) -> None:
        """Evaluate a model

        Args:
            params (SKTaskParams): Task parameters

        """
        raise NotImplementedError

    @staticmethod
    def export(params: SKTaskParams) -> None:
        """Export a model

        Args:
            params (SKTaskParams): Task parameters

        """
        raise NotImplementedError

    @staticmethod
    def demo(params: SKTaskParams) -> None:
        """Run a demo

        Args:
            params (SKTaskParams): Task parameters

        """
        raise NotImplementedError