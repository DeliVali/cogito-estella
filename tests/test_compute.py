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


def test_graph_decoder_has_no_length_factor():
    # el graph decoder es O(1): su costo NO escala con tokens_per_concept
    bd = c.concept_graph_training_flops(n_ct=0, n_gdec=1000, n_enc=0, n_concepts=10,
                                        tokens_per_concept=25, n_epochs=1)
    assert bd.sonar_decoder == 6 * 1000 * 10   # sin factor 25
    bd2 = c.concept_graph_training_flops(n_ct=0, n_gdec=1000, n_enc=0, n_concepts=10,
                                         tokens_per_concept=100, n_epochs=1)
    assert bd2.sonar_decoder == bd.sonar_decoder  # invariante a la longitud del texto


def test_graph_decoder_much_cheaper_than_sonar_text():
    # inferencia: graph decoder de 5M vs SONAR texto 605M×25 tokens
    gr = c.concept_graph_inference_flops_per_concept(n_ct=100e6, n_gdec=5e6)
    txt = 2 * 100e6 + 2 * 605e6 * 25   # concepto + SONAR texto
    assert gr < txt / 50   # al menos 50× más barato por concepto


def test_context_processing_favors_compression():
    # comprimir el contexto ~L_tok× reduce el costo de prefill (lineal y cuadrático)
    tokens, dim, L = 4096, 1024, 25
    tok_cost = c.context_processing_flops(n_model=100_000_000, n_positions=tokens, dim=dim)
    con_cost = c.context_processing_flops(n_model=100_000_000, n_positions=tokens // L, dim=dim)
    assert con_cost < tok_cost
