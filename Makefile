# Usage:
# Open a new terminal window, then `make <command>`.

clean-build:
# Build the front end, and output files to the static folder of the Python server
	rm --force --recursive server/server/static
	mkdir server/server/static
	rm --force --recursive ui/.parcel-cache
	npx parcel build ui/src/index.html --no-scope-hoist --dist-dir server/server/static

start-ui:
# Start UI in development mode
	npx parcel ui/src/index.html --dist-dir ui/dist

start-server:
# Start the Python Quart server
	python server/server/app.py

parcel-clear-cache:
# Clear the Parcel cache (can resolve some issues)
	rm --force --recursive ui/.parcel-cache