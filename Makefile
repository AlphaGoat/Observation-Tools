# Makefile — build, deploy, and operate the Observation-Tools pipeline on minikube.
#
# Quick start
# -----------
#   make minikube-start   # start minikube (first time only)
#   make build            # build all Docker images inside minikube's daemon
#   make deploy           # apply all Kubernetes manifests
#   make wait             # wait for every pod to become Ready
#   make url              # print the coordinator's external URL
#
# Daily workflow
# --------------
#   make logs-coordinator # tail coordinator logs
#   make status           # show pod health across the namespace
#   make port-forward     # local :8080 → coordinator service (alternative to NodePort)
#
# Teardown
# --------
#   make undeploy         # delete all manifests (keeps minikube running)
#   make minikube-stop    # stop minikube

NAMESPACE   := observation-tools
REGISTRY    :=                          # leave empty for local minikube builds
IMAGE_TAG   := latest

# Image names — prefix with $(REGISTRY)/ in production
IMG_EXTRACTOR   := observation-tools/source-extractor:$(IMAGE_TAG)
IMG_SOLVER      := observation-tools/plate-solver:$(IMAGE_TAG)
IMG_PROJECTOR   := observation-tools/projector:$(IMAGE_TAG)
IMG_ASSOCIATOR  := observation-tools/associator:$(IMAGE_TAG)
IMG_COORDINATOR := observation-tools/pipeline-coordinator:$(IMAGE_TAG)

# Dockerfiles
DF_EXTRACTOR   := src/source_extraction/Dockerfile
DF_SOLVER      := src/astrometry/Dockerfile
DF_PROJECTOR   := src/obs/Dockerfile
DF_ASSOCIATOR  := src/associator/Dockerfile
DF_COORDINATOR := src/pipeline/Dockerfile

.PHONY: all minikube-start minikube-stop build \
        build-extractor build-solver build-projector build-associator build-coordinator \
        deploy undeploy wait status url port-forward \
        logs-coordinator logs-solver logs-star logs-satellite logs-projector logs-associator \
        populate-index shell-solver test-health

# ── Minikube ──────────────────────────────────────────────────────────────────

minikube-start:
	minikube start \
	    --cpus=4 \
	    --memory=8192 \
	    --disk-size=30g \
	    --driver=docker
	minikube addons enable metrics-server

minikube-stop:
	minikube stop

# ── Build ─────────────────────────────────────────────────────────────────────
# All images are built inside minikube's Docker daemon so Kubernetes can pull
# them without a registry.  `eval $(minikube docker-env)` is injected per-rule.

build: build-extractor build-solver build-projector build-associator build-coordinator

build-extractor:
	@echo "▶ Building source-extractor…"
	eval $$(minikube docker-env) && \
	    docker build -f $(DF_EXTRACTOR) -t $(IMG_EXTRACTOR) .

build-solver:
	@echo "▶ Building plate-solver…"
	eval $$(minikube docker-env) && \
	    docker build -f $(DF_SOLVER) -t $(IMG_SOLVER) .

build-projector:
	@echo "▶ Building projector…"
	eval $$(minikube docker-env) && \
	    docker build -f $(DF_PROJECTOR) -t $(IMG_PROJECTOR) .

build-associator:
	@echo "▶ Building associator…"
	eval $$(minikube docker-env) && \
	    docker build -f $(DF_ASSOCIATOR) -t $(IMG_ASSOCIATOR) .

build-coordinator:
	@echo "▶ Building pipeline-coordinator…"
	eval $$(minikube docker-env) && \
	    docker build -f $(DF_COORDINATOR) -t $(IMG_COORDINATOR) .

# ── Deploy ────────────────────────────────────────────────────────────────────

deploy:
	kubectl apply -k k8s/
	@echo ""
	@echo "Manifests applied.  Run 'make wait' to block until all pods are Ready."

undeploy:
	kubectl delete -k k8s/ --ignore-not-found

wait:
	@echo "Waiting for all Deployments in $(NAMESPACE) to become Available…"
	kubectl wait deployment \
	    star-extractor satellite-extractor plate-solver projector associator pipeline-coordinator \
	    -n $(NAMESPACE) \
	    --for=condition=Available \
	    --timeout=300s
	@echo "All pods are Ready."

# ── Populate index volume ─────────────────────────────────────────────────────
# Run this after 'make wait' to load your pre-built index files into the PVC.
# Adjust SOURCE_INDEX_DIR to point at your local indices/ directory.
SOURCE_INDEX_DIR ?= indices/

populate-index:
	@echo "Copying $(SOURCE_INDEX_DIR) into plate-solver pod at /indices/ …"
	$(eval POD := $(shell kubectl get pods -n $(NAMESPACE) -l app=plate-solver \
	    -o jsonpath='{.items[0].metadata.name}'))
	@test -n "$(POD)" || (echo "ERROR: no plate-solver pod found" && exit 1)
	kubectl -n $(NAMESPACE) cp $(SOURCE_INDEX_DIR) $(POD):/indices/
	@echo "Done.  Restart the plate-solver to reload:"
	@echo "  kubectl rollout restart deployment/plate-solver -n $(NAMESPACE)"

# ── Inspect ───────────────────────────────────────────────────────────────────

status:
	kubectl get pods,svc -n $(NAMESPACE) -o wide

url:
	@echo "Pipeline coordinator URL:"
	minikube service pipeline-coordinator -n $(NAMESPACE) --url

port-forward:
	@echo "Forwarding localhost:8080 → pipeline-coordinator:8080 …"
	@echo "(Ctrl-C to stop)"
	kubectl port-forward -n $(NAMESPACE) svc/pipeline-coordinator 8080:8080

# ── Logs ─────────────────────────────────────────────────────────────────────

logs-coordinator:
	kubectl logs -n $(NAMESPACE) -l app=pipeline-coordinator -f --tail=100

logs-solver:
	kubectl logs -n $(NAMESPACE) -l app=plate-solver -f --tail=100

logs-star:
	kubectl logs -n $(NAMESPACE) -l app=star-extractor -f --tail=100

logs-satellite:
	kubectl logs -n $(NAMESPACE) -l app=satellite-extractor -f --tail=100

logs-projector:
	kubectl logs -n $(NAMESPACE) -l app=projector -f --tail=100

logs-associator:
	kubectl logs -n $(NAMESPACE) -l app=associator -f --tail=100

# ── Debug ─────────────────────────────────────────────────────────────────────

shell-solver:
	$(eval POD := $(shell kubectl get pods -n $(NAMESPACE) -l app=plate-solver \
	    -o jsonpath='{.items[0].metadata.name}'))
	kubectl exec -it -n $(NAMESPACE) $(POD) -- /bin/bash

# ── Quick health smoke test ───────────────────────────────────────────────────
# Calls /health on the coordinator (which probes all downstream services).

test-health:
	$(eval COORD_URL := $(shell minikube service pipeline-coordinator \
	    -n $(NAMESPACE) --url 2>/dev/null))
	@test -n "$(COORD_URL)" || (echo "ERROR: could not get coordinator URL" && exit 1)
	@echo "Probing $(COORD_URL)/health …"
	curl -s $(COORD_URL)/health | python3 -m json.tool
