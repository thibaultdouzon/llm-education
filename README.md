# LLM Education

Small lib implementing LLMs for educational purposes

## Objectives

- [x] Implement Transformer architecture
- [x] Load models from Huggingface and reproduce results
- [x] Generation with sampling, beam search
- [ ] Transition to jax / equinox
    - fix all layers, implement basic tests
    - Remove batch and replace with vmap (same with length)
    
- [ ] Tokenizers
- [ ] Dataset (Shakespeare?)
- [ ] Basic fine tuning
- [ ] Better Transformer++ (from mamba)
- [ ] KV cache for inference
- [ ] Accelerate inference with smaller predictor
- [ ] Generation guided by grammar
- [ ] SSM / Mamba architecture
- [ ] Visual Transformers
- [ ] Efficient models
- [ ] Triton?
