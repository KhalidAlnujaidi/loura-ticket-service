PY := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)

.PHONY: install install-slim seed run test eval synthetic finetune

install:
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt -r requirements-laya.txt

install-slim:
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

seed:
	$(PY) -m app.seed

run:
	$(PY) -m uvicorn app.main:app --reload

test:
	$(PY) -m pytest

# Agreement report over the labelled tickets (data/labelled_tickets.json +
# data/synthetic_tickets.json when present). MODEL=<id|dir> to score a
# fine-tuned checkpoint instead of the base one.
eval:
	$(PY) -m scripts.eval_agreement --backend laya $(if $(MODEL),--model $(MODEL)) \
		$(if $(ONLY),--only $(ONLY)) $(if $(SPLIT),--synthetic-split $(SPLIT)) \
		--out reports/eval-$(if $(MODEL),$(notdir $(MODEL)),base)$(if $(ONLY),-$(ONLY)).json

# Grow the labelled set with cross-verified synthetic tickets from the free pool.
# Needs OPENROUTER_API_KEY and a fresh ~/.cache/or-swarm-status.json (probe first).
synthetic:
	$(PY) -m scripts.generate_synthetic --per-cell $(or $(PER_CELL),8)

# Fine-tune Laya on the labelled tickets (single device; MPS on Apple silicon).
finetune:
	$(PY) -m scripts.finetune_laya --out $(or $(OUT),checkpoints/loura-tickets-v1) \
		$(if $(EXCLUDE),--exclude-ids $(EXCLUDE))
