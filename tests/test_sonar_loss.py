import pytest
import torch

pytestmark = pytest.mark.integration


def test_ce_discriminates_true_vs_random():
    """El embedding verdadero de una oración debe dar CE mucho menor que uno aleatorio."""
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

    from cogito_estella.model.sonar_loss import SonarCELoss

    celoss = SonarCELoss(device="cpu")
    enc = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder",
        device=torch.device("cpu"))
    sent = "Machine learning models require careful evaluation."
    true_emb = enc.predict([sent], source_lang="eng_Latn")  # [1,1024]

    with torch.no_grad():
        ce_true = celoss.loss(true_emb, [sent], "eng_Latn").item()
        ce_rand = celoss.loss(torch.randn_like(true_emb), [sent], "eng_Latn").item()
    assert ce_true < 2.0, f"CE del embedding verdadero demasiado alta: {ce_true}"
    assert ce_rand > 10.0, f"CE del aleatorio demasiado baja: {ce_rand}"
    assert ce_true < ce_rand / 5


def test_gradient_flows_to_embedding_but_not_decoder():
    from cogito_estella.model.sonar_loss import SonarCELoss

    celoss = SonarCELoss(device="cpu")
    emb = torch.randn(2, 1024, requires_grad=True)
    loss = celoss.loss(emb, ["hello world there", "another test sentence"], "eng_Latn")
    loss.backward()
    assert emb.grad is not None and emb.grad.abs().sum() > 0
    # el decoder queda congelado: ningún parámetro acumula gradiente
    assert all(p.grad is None for p in celoss.model.parameters())
