"""Shared singleton dependencies for API routes."""

from backend.app.experiment_loader.loader import ExperimentArtifactLoader
from backend.app.simulation.runner import simulation_service
from experiments.phase6_model_comparison.episode_library.library import EpisodeLibrary

artifact_loader = ExperimentArtifactLoader()
episode_library = EpisodeLibrary()


def get_loader() -> ExperimentArtifactLoader:
    return artifact_loader


def get_episode_library() -> EpisodeLibrary:
    return episode_library


def get_sim_service():
    return simulation_service
