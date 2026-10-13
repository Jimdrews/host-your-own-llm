"""Model lookups on the Hugging Face Hub."""

from huggingface_hub import auth_check, model_info
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from ..domain import ModelCheck
from ..errors import ModelError


class HubClient:
    def check_model(self, model_id: str, token: str | None) -> ModelCheck:
        """Look up a model's weights, or raise ModelError if it can't be served."""
        try:
            info = model_info(model_id, token=token or None, files_metadata=True)
        except GatedRepoError:  # a subclass of RepositoryNotFoundError, so it must come first
            raise ModelError(f"{model_id} is gated and this token doesn't have access")
        except RepositoryNotFoundError:
            raise ModelError(f"{model_id} was not found on Hugging Face (or it is private and needs a token)")
        except HfHubHTTPError as e:
            raise ModelError(f"Couldn't look up {model_id}: {e}")

        weights = [s for s in info.siblings or [] if s.rfilename.endswith(".safetensors")]
        if not weights:
            raise ModelError(f"{model_id} has no .safetensors weights, which the S3 streamer requires")
        return ModelCheck(model_id=model_id, size_bytes=sum(s.size or 0 for s in weights), gated=bool(info.gated))

    def has_access(self, model_id: str, token: str) -> bool:
        try:
            auth_check(model_id, token=token)
            return True
        except (GatedRepoError, RepositoryNotFoundError):
            return False
