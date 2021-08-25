// Significantly adapted from Leaflet.SelectAreaFeature.js
// GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007, commit: 0bc14e1 30 Jan
// https://github.com/sandropibia/Leaflet.SelectAreaFeature
//
// Additions:
// - Add options -> fillOpacity: .35 and added to _doMouseUp -> L.polygon()
// - Add getAreas()
// - Add comments
// - Add options -> paneName (must add polygon to pane, otherwise doesn't show if using Leaflet panes).
//		Searched for 'addTo(' to see where lines / polygons are added to the map, and added pane option.
//		Ensure <Pane /> has been defined first before initialising this.
// - Add smoothFactor in _doMouseUp -> L.polygon(). Reduces the number of points in order to increase 
//		performance when determining which points fall inside the polygon on the server-side.
// - Removed 
//		- getFeaturesSelected()
//		- isMarkerInsidePolygon
//		- mouseup, mousedown and mousemove handlers (will be handled by mapController)

(function (factory) {
	if (typeof define === 'function' && define.amd) {
		// AMD
		define(['leaflet'], factory);
	} else if (typeof module !== 'undefined') {
		// Node/CommonJS
		module.exports = factory(require('leaflet'));
	} else {
		// Browser globals
		if (typeof window.L === 'undefined') {
			throw new Error('Leaflet must be loaded first');
		}
		factory(window.L);
	}
}(function (L) {
	"use strict";
	L.SelectAreaFeature = L.Handler.extend({

		options: {
			color: 'green',
			fillOpacity: 0.01,
			weight: '1.5',
			dashArray: '5, 5, 1, 5',
			selCursor: 'crosshair',
			normCursor: '',
			paneName: 'area-select-pane' // CS
		},

		initialize: function (map, options) {
			this._map = map;

			this._pre_latlon = '';
			this._post_latlon = '';
			this._ARR_latlon_line = [];
			this._ARR_latlon = [];
			this._flag_new_shape = false;
			this._area_pologon_layers = [];

			this._area_line = '';
			this._area_line_new = '';

			L.setOptions(this, options);
		},

		addHooks: function () {
			//this._map.on('mousedown', this._doMouseDown, this);
			//this._map.on('mouseup', this._doMouseUp, this);
			this._map.dragging.disable();
			this._map._container.style.cursor = this.options.selCursor;
		},

		removeHooks: function () {
			//this._map.off('mousemove');
			//this._map.off('mousedown');
			//this._map.off('mouseup');
			this._map._container.style.cursor = this.options.normCursor;
			this._map.dragging.enable();
		},

		onDrawEnd: null,

		_doMouseUp: function (ev) {

			this._pre_latlon = '';
			this._post_latlon = '';
			this._ARR_latlon_line = [];
			if (this._flag_new_shape) {
				//this._area_pologon_layers.push(L.polygon(this._ARR_latlon, {color: this.options.color}).addTo(this._map)); // CS
				this._area_pologon_layers.push(
					L.polygon(this._ARR_latlon, { color: this.options.color, smoothFactor: 0.75, weight: this.options.weight, fillOpacity: this.options.fillOpacity, pane: this.options.paneName }
					).addTo(this._map)); // CS

				if (this._map.hasLayer(this._area_line)) {
					this._map.removeLayer(this._area_line);
				}
				if (this._map.hasLayer(this._area_line_new)) {
					this._map.removeLayer(this._area_line_new);
				}
				this._flag_new_shape = false;
			}
			//this._map.off('mousemove');
			if (this.onDrawEnd) this.onDrawEnd();
		},

		onDrawStart: null,

		_doMouseDown: function (ev) {
			if (this.onDrawStart) this.onDrawStart();

			this._ARR_latlon = [];
			this._flag_new_shape = true;
			this._area_pologon = '';
			this._area_line_new = '';
			this._area_line = '';
			//this._map.on('mousemove', this._doMouseMove, this);
		},

		_doMouseMove: function (ev) {
			this._ARR_latlon.push(ev.latlng);
			if (this._pre_latlon == '' || this._pre_latlon == "undefined") {
				this._pre_latlon = ev.latlng;
				this._ARR_latlon_line.push(this._pre_latlon);
			}
			else if (this._pre_latlon != '' && (this._post_latlon == '' || this._post_latlon == "undefined")) {
				this._post_latlon = ev.latlng;
				this._ARR_latlon_line.push(this._post_latlon);
			}
			else {
				this._pre_latlon = this._post_latlon;
				this._post_latlon = ev.latlng;
				this._ARR_latlon_line.push(this._pre_latlon);
				this._ARR_latlon_line.push(this._post_latlon);
			}

			if (this._pre_latlon != '' && this._post_latlon != '') {
				if (this._area_line_new == '' && this._area_line == '') {
					this._area_line = L.polyline(this._ARR_latlon_line, {
						color: this.options.color,
						weight: this.options.weight,
						dashArray: this.options.dashArray,
						pane: this.options.paneName
					});

					this._area_line.addTo(this._map);
				}
				if (this._area_line_new == '' && this._area_line != '') {
					this._area_line_new = L.polyline(this._ARR_latlon_line, {
						color: this.options.color,
						weight: this.options.weight,
						dashArray: this.options.dashArray,
						pane: this.options.paneName
					});

					this._area_line_new.addTo(this._map);
					this._map.removeLayer(this._area_line);
				}
				if (this._area_line_new != '' && this._area_line != '') {
					this._area_line = L.polyline(this._ARR_latlon_line, {
						color: this.options.color,
						weight: this.options.weight,
						dashArray: this.options.dashArray,
						pane: this.options.paneName
					});
					this._area_line.addTo(this._map);
					this._map.removeLayer(this._area_line_new);
					this._area_line_new = '';
				}
			}
		},

		getAreaLatLng: function () {
			// Array of polygon points, {Lat: Lng:}
			return this._ARR_latlon;
		},

		removeAllArea: function () {
			// Remove all polygon layers
			var _i = 0;
			while (_i < this._area_pologon_layers.length) {
				this._map.removeLayer(this._area_pologon_layers[_i]);
				_i++;
			}
			this._area_pologon_layers.splice(0, _i);
		},

		removeLastArea: function () {
			// Remove last polygon layer
			var index = this._area_pologon_layers.length - 1;
			this._map.removeLayer(this._area_pologon_layers[index]);
			this._area_pologon_layers.splice(index, 1);
		},

		getAreas: function () {
			// Return all polygon layers
			return this._area_pologon_layers;
		}

	});

}, window));

L.Map.addInitHook('addHandler', 'selectAreaFeature', L.SelectAreaFeature);
