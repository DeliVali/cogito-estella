from cogito_estella import compute as c


def test_frozen_decoder_uses_factor_4_not_6():
    # decoder congelado: 4·N·D (no 6), porque no computa grad de pesos.
    bd = c.concept_training_flops(n_ct=0, n_dec=1000, n_enc=0, n_concepts=10,
                                  tokens_per_concept=5, n_epochs=1)
    # tokens decodificados = 10*1*5 = 50; 4 * 1000 * 50 = 200_000
    assert bd.sonar_decoder == 4 * 1000 * 50
    assert bd.concept_transformer == 0


def test_concept_transformer_uses_factor_6():
    bd = c.concept_training_flops(n_ct=500, n_dec=0, n_enc=0, n_concepts=10,
                                  tokens_per_concept=5, n_epochs=2)
    # conceptos procesados = 10*2 = 20; 6 * 500 * 20
    assert bd.concept_transformer == 6 * 500 * 20


def test_encoder_amortized_paid_once():
    # el encode se paga una vez (independiente de n_epochs en el término devuelto)
    bd1 = c.concept_training_flops(0, 0, n_enc=1000, n_concepts=10, tokens_per_concept=5, n_epochs=1)
    bd5 = c.concept_training_flops(0, 0, n_enc=1000, n_concepts=10, tokens_per_concept=5, n_epochs=5)
    assert bd1.sonar_encoder_amortized == bd5.sonar_encoder_amortized == 2 * 1000 * 50


def test_token_training_flops():
    bd = c.token_training_flops(n_tt=500, n_concepts=10, tokens_per_concept=5, n_epochs=2)
    # tokens = 10*5*2 = 100; 6*500*100
    assert bd.token_transformer == 6 * 500 * 100


def test_concept_inference_dominated_by_decoder():
    # con N_dec grande, el costo por token de inferencia lo domina el decoder
    per_tok = c.concept_inference_flops_per_token(n_ct=1_000_000, n_dec=600_000_000,
                                                  tokens_per_concept=25)
    assert per_tok > 2 * 600_000_000  # >= 2·N_dec
    assert per_tok < 2 * 600_000_000 + 2 * 1_000_000  # + término pequeño del CT


def test_context_processing_favors_compression():
    # comprimir el contexto ~L_tok× reduce el costo de prefill (lineal y cuadrático)
    tokens, dim, L = 4096, 1024, 25
    tok_cost = c.context_processing_flops(n_model=100_000_000, n_positions=tokens, dim=dim)
    con_cost = c.context_processing_flops(n_model=100_000_000, n_positions=tokens // L, dim=dim)
    assert con_cost < tok_cost
