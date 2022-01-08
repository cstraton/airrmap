# Usage:
# Open a new terminal window, then `make <command>`.

.PHONY : \
	start-server \
	start-ui \
	parcel-clear-cache \
	docker-run \
	docker-buildrun \
	docker-build \
	clean-build \
	init-data-folder

# Constants
PORT=5000
CONTAINER_NAME=airrmap-server
IMAGE_NAME=airrmap-server
DATA_FOLDER_VAR=AIRRMAP_DATA


# --- These commands must be run from the development environment ---
# --- (requires npx, parcel etc. ...)							  ---

# Build the front end, and output files to the static folder of the Python server
clean-build:
	rm --force --recursive server/server/static
	mkdir server/server/static
	rm --force --recursive ui/.parcel-cache
	npx parcel build ui/src/index.html --no-scope-hoist --dist-dir server/server/static

# Start UI in development mode
start-ui:
	npx parcel ui/src/index.html --dist-dir ui/dist

# Start the Python Quart server
start-server:
	python server/server/app.py


# --- These commands can be run from Terminal ---

# Build Docker application server
docker-build:
	docker build \
		--file=./Dockerfile \
		--tag=$(IMAGE_NAME) \
		.


# Run Docker application server
# Ensure AIRRMAP_DATA environment variable folder has been set beforehand
docker-run:

# Check if container exists, remove otherwise errors on next step
# REF: https://stackoverflow.com/questions/38576337/how-to-execute-a-bash-command-only-if-a-docker-container-with-a-given-name-does
ifneq ($(shell docker ps -aq -f status=exited -f name=$(CONTAINER_NAME)),)
	docker rm $(CONTAINER_NAME)
endif

# Run the container
	docker run \
		-v $(AIRRMAP_DATA):/airrmap-data \
		--publish=$(PORT):$(PORT) \
		--name=$(CONTAINER_NAME) \
		$(IMAGE_NAME)


# Build and run Docker application server
docker-buildrun : docker-build docker-run

# Clear the Parcel cache (can resolve some issues)
parcel-clear-cache:
	rm --force --recursive ui/.parcel-cache


# Create data folder if it doesn't exist,
# and copy necessary files.
# Environment variable AIRRMAP_DATA should be set first.
init-data-folder:

# Check variable is set
ifeq ($($(DATA_FOLDER_VAR)),)
	$(error '$(DATA_FOLDER_VAR) environment variable not set.')
endif

# Create folder if it doesn't exist
	mkdir -p "$($(DATA_FOLDER_VAR))"

# Copy files to data folder
	@cp ./data/appconfig.yaml "$($(DATA_FOLDER_VAR))"
	@echo Copied appconfig.yaml to "$($(DATA_FOLDER_VAR))"
	@cp ./data/envconfig.yaml "$($(DATA_FOLDER_VAR))"
	@echo Copied envconfig.yaml to "$($(DATA_FOLDER_VAR))"
	@cp ./data/envinit.sh "$($(DATA_FOLDER_VAR))"
	@echo Copied envinit.sh to "$($(DATA_FOLDER_VAR))"
	@echo 'Completed, see "$($(DATA_FOLDER_VAR))"'