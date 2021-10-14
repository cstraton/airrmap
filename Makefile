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


# vars
PORT=5000
CONTAINER_NAME=airrmap-server
IMAGE_NAME=airrmap-server
DATA_FOLDER_VAR=AIRRMAP_DATA

# Build the front end, and output files to the static folder of the Python server
clean-build:
	rm --force --recursive server/server/static
	mkdir server/server/static
	rm --force --recursive ui/.parcel-cache
	npx parcel build ui/src/index.html --no-scope-hoist --dist-dir server/server/static

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

# Start UI in development mode
start-ui:
	npx parcel ui/src/index.html --dist-dir ui/dist

# Start the Python Quart server
start-server:
	python server/server/app.py


# Set env variable for data folder:
# (if not already set)
init-data-folder:
ifeq ($(fdr),)
	@echo '"fdr" argument not supplied. Run: make init-data-folder fdr="your/data/path"'
else
ifeq ($($(DATA_FOLDER_VAR)),)
	@echo '$(DATA_FOLDER_VAR) variable is not set. Will add to "$(HOME)/.profile"'
	echo '' >> $(HOME)/.profile
	echo '# AIRRMAP Data Folder' >> $(HOME)/.profile
	echo 'export $(DATA_FOLDER_VAR)=$(fdr)' >> $(HOME)/.profile
	. $(HOME)/.profile
	export $(DATA_FOLDER_VAR)=$(fdr)
	@echo 'Added to "$(HOME)/.profile"'
else	
	@echo '$(DATA_FOLDER_VAR) variable is already set to "$($(DATA_FOLDER_VAR))"'
	@echo 'Will NOT be added to "$(HOME)/.profile"'
endif
endif
# Copy config.yaml to data folder
	cp ./server/data/example/config.yaml $($(DATA_FOLDER_VAR))
	@echo Copied config.yaml to $($(DATA_FOLDER_VAR))