# Disha — one-command developer workflow.
# Targets:
#   make install     install python + frontend deps
#   make build       rebuild all frozen data artifacts (twin, signals, uplift, OPE)
#   make plans       precompute dual-arm route plans for the demo date
#   make api         run FastAPI on :8000
#   make web         run Vite dev server on :5173 (proxies /api -> :8000)
#   make demo        the everything: build artifacts + start api + start web
#   make test        run the pytest suite
#   make clean       remove built artifacts (keeps raw data)
.PHONY: install build plans api web demo test clean

PY := python

install:
	$(PY) -m pip install -e .
	cd app/web && npm install --no-audit --no-fund

build:
	$(PY) -m disha.twin.build
	$(PY) -m disha.signals.run
	$(PY) -m disha.uplift.train
	$(PY) -m disha.eval.residualized_qini
	$(PY) -m disha.ope.evaluate
	$(PY) -m disha.eval.business_case
	$(PY) -m disha.optimizer.router

plans:
	$(PY) -m disha.optimizer.router

api:
	$(PY) -m uvicorn disha.api.main:app --host 0.0.0.0 --port 8000 --reload

web:
	cd app/web && npm run dev

# Best-effort one-command demo. Open http://localhost:5173 once both processes are listening.
demo:
	@echo ">> Building frozen artifacts (skip if already built)..."
	@if [ ! -f data/processed/business_case.json ]; then $(MAKE) build; fi
	@echo ">> Starting API + Web. Open http://localhost:5173 ."
	@( $(PY) -m uvicorn disha.api.main:app --host 0.0.0.0 --port 8000 & echo $$! > .api.pid ) \
	&& ( cd app/web && npm run dev & echo $$! > ../../.web.pid ) \
	&& trap 'kill `cat .api.pid` `cat .web.pid` 2>/dev/null; rm -f .api.pid .web.pid' EXIT \
	&& wait

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/processed/plans
	rm -f  data/processed/business_case.json data/processed/ope.json
	rm -f  data/processed/uplift_eval.json data/processed/uplift_*_cate.parquet
