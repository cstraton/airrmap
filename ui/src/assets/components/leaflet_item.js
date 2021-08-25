// Leaflet Map component

// Imports
import { Loader } from 'semantic-ui-react';
import React, { useState, useCallback, useEffect, useMemo, useRef } from "react";
import { MapContainer, TileLayer, ImageOverlay, Pane, useMap } from 'react-leaflet';
import '../js/vendor/leaflet.area_select/leaflet.area_select.js'; // Area selection
import '../js/vendor/L.SimpleGraticule/L.SimpleGraticule.js'; // Grid of coordinates
import '../js/vendor/L.SimpleGraticule/L.SimpleGraticule.css';
import { CRS } from 'leaflet';
//import { MinimapControl } from './leaflet_minimap.js';
import 'leaflet/dist/leaflet.css';
import imgMarkerShadow from 'leaflet/dist/images/marker-shadow.png';
//import imgMarkerIcon from 'leaflet/dist/images/marker-icon.png';
import imgMarkerIcon from '../img/leaflet-color-markers/marker-icon-blue.png';
import L from 'leaflet';
import 'leaflet.markercluster';
import './css/leaflet_item.css';

// Vars
const mapTheme = 'leaflet-container-light';

// Initial map position. mid-tile (256x256).
// Subtract 10 for lat, due to map label.
// lat (up/down), lng (left/right)
const initialPosition = [118, 128];

function LeafletItem({
  id, // need it as a prop, 'key' will be undefined if trying to read.
  mapController,
  mapLabel,
  iRow,
  iCol,
  facetRowValue,
  facetColValue,
  mapStatsEnabled,
  stats }) {

  // Final render display
  function DisplayMap({ id, mapController, mapTheme, facetRowValue, facetColValue, mapLabel, stats }) {

    // State - keep contained in this function (otherwise flicker issue due to setMap).
    const [map, _setMap] = useState(null);
    const [areaSelect, setAreaSelect] = useState(null);
    const [graticule, setGraticule] = useState(null);
    const [KDELoading, setKDELoading] = useState(false);
    const [BinnedLoading, setBinnedLoading] = useState(false);

    // Map setter
    // Set both the map and area selection instance
    const setMap = ((map) => {

      // Set max zoom to avoid error when using
      // MarkerClusterGroup in LayerMarkers()
      // REF: https://github.com/Leaflet/Leaflet.markercluster/issues/611
      map._layersMaxZoom = 10;

      _setMap(map);
      const _areaSelect = map.selectAreaFeature.enable(); // new instance.
      _areaSelect.disable(); // turn off selection mode.
      setAreaSelect(_areaSelect);
    });

    // useMemo to only render first time.
    const mapContainer = useMemo(() => (

      <MapContainer
        key={id}
        className={mapTheme}
        center={initialPosition}
        zoom={0}
        boxZoom={false}
        zoomControl={false}
        zoomAnimation={true}
        zoomSnap={0}
        zoomDelta={1}
        wheelPxPerZoomLevel={60/1}
        inertia={true}
        attributionControl={false}
        fadeAnimation={true}
        scrollWheelZoom={true}
        crs={CRS.Simple}
        smFacetRowValue={facetRowValue} // Store props in map container. Used for facet roi queries.
        smFacetColValue={facetColValue} // sm prefix to prevent conflicts with Leaflet.
        whenCreated={setMap} // Store map instance
      >
        <LayerBinnedTile mapController={mapController} facetRowValue={facetRowValue} facetColValue={facetColValue} setBinnedLoading={setBinnedLoading} />
        <LayerKDE mapController={mapController} facetRowValue={facetRowValue} facetColValue={facetColValue} setKDELoading={setKDELoading} />
        <LayerAreaSelect />
        <LayerGridCoordinates setGraticule={setGraticule} />

        {/* Map Label */}
        <span className={'map-label'}>{mapLabel}</span>

      </MapContainer>
    ), [facetRowValue, facetColValue]);

    // Return final render
    return (
      <React.Fragment>
        {mapContainer}
        <LayerStats mapStatsEnabled={mapStatsEnabled} stats={stats} />
        {/*{map ? <LayerPositionInfo map={map} /> : null}*/}
        {map ? <LayerMarkers mapController={mapController} map={map} /> : null}
        {map ? <MapEventsController map={map} mapController={mapController} areaSelect={areaSelect} graticule={graticule} /> : null}
        <Loader active={KDELoading || BinnedLoading} />
      </React.Fragment>
    );
  }

  // Handle events from the map and area selection
  function MapEventsController({ map, mapController, areaSelect, graticule }) {

    // Handle mousedown
    // Pass on mouse down event
    const onMouseDown = useCallback((ev) => {
      mapController.onMouseDown(ev)
    }, [map, mapController]);
    useEffect(() => {
      map.on('mousedown', onMouseDown)
      return () => {
        map.off('mousedown', onMouseDown)
      }
    }, [map, onMouseDown]);

    // Handle mouseup
    // Pass on mouse up event
    const onMouseUp = useCallback((ev) => {
      mapController.onMouseUp(ev)
    }, [map, mapController]);
    useEffect(() => {
      map.on('mouseup', onMouseUp)
      return () => {
        map.off('mouseup', onMouseUp)
      }
    }, [map, onMouseUp]);

    // Handle mouseover (map enter)
    // Set active map
    const onMouseOver = useCallback((ev) => {
      mapController.setActiveMap(map);
    }, [map, mapController]);
    useEffect(() => {
      map.on('mouseover', onMouseOver)
      return () => {
        map.off('mouseover', onMouseOver)
      }
    }, [map, onMouseOver]);

    // Handle mousemove
    // Pass on mouse move event
    const onMouseMove = useCallback((ev) => {
      mapController.onMouseMove(ev)
    }, [map, mapController]);
    useEffect(() => {
      map.on('mousemove', onMouseMove)
      return () => {
        map.off('mousemove', onMouseMove)
      }
    }, [map, onMouseMove]);

    // If this is the active map, pass on move events
    const onMove = useCallback((ev) => {
      if (map === mapController.activeMap) {
        mapController.onMove(ev);
      }
    }, [map, mapController])
    useEffect(() => {
      map.on('move', onMove)
      return () => {
        map.off('move', onMove)
      }
    }, [map, onMove]);

    // If this is not the active map, set view to the active map
    const onMasterMove = useCallback(() => {
      if (map !== mapController.activeMap) {
        if (mapController.activeMap) {
          const center = mapController.activeMap.getCenter();
          const zoom = mapController.activeMap.getZoom();
          map.setView(center, zoom, {
            'animate': false
          });
        }
      }
    })
    useEffect(() => {
      mapController.addEventListener('moved', onMasterMove);
      return () => {
        mapController.removeEventListener('moved', onMasterMove);
      }
    }, [mapController, onMasterMove])


    // --- AREA SELECTION ---

    // Enable area select (on all, in case we want to draw on any)
    const onAreaSelectEnabled = useCallback(() => {
      areaSelect.enable();
    })
    useEffect(() => {
      mapController.addEventListener('areaselectenabled', onAreaSelectEnabled);
      return () => {
        mapController.removeEventListener('areaselectenabled', onAreaSelectEnabled);
      }
    }, [mapController, onAreaSelectEnabled])

    // Disable area select
    const onAreaSelectDisabled = useCallback(() => {
      areaSelect.disable();
    })
    useEffect(() => {
      mapController.addEventListener('areaselectdisabled', onAreaSelectDisabled);
      return () => {
        mapController.removeEventListener('areaselectdisabled', onAreaSelectDisabled);
      }
    }, [mapController, onAreaSelectDisabled])

    // Remove last selection if active map or map sync is on
    const onAreaSelectRemoveLast = useCallback(() => {
      if (map === mapController.activeMap || mapController.mapSync) {
        areaSelect.removeLastArea();
      }
    })
    useEffect(() => {
      mapController.addEventListener('areaselectremovelast', onAreaSelectRemoveLast);
      return () => {
        mapController.removeEventListener('areaselectremovelast', onAreaSelectRemoveLast);
      }
    }, [mapController, onAreaSelectRemoveLast])

    // Clear selections
    const onAreaSelectClear = useCallback(() => {
      areaSelect.removeAllArea();
    })
    useEffect(() => {
      mapController.addEventListener('areaselectclear', onAreaSelectClear);
      return () => {
        mapController.removeEventListener('areaselectclear', onAreaSelectClear);
      }
    }, [mapController, onAreaSelectClear])

    // Mouse Down
    // If area select is enabled, this is the active map or map sync is on
    const onMasterMouseDown = useCallback((ev) => {
      if (mapController.areaSelectEnabled) {
        if (map === mapController.activeMap || mapController.mapSync) {
          areaSelect._doMouseDown(ev.detail)
        }
      }

    });
    useEffect(() => {
      mapController.addEventListener('mousedown', onMasterMouseDown);
      return () => {
        mapController.removeEventListener('mousedown', onMasterMouseDown);
      }
    }, [mapController, onMasterMouseDown])

    // Mouse Move
    // If area select is enabled, this is the active map or map sync is on
    const onMasterMouseMove = useCallback((ev) => {
      if (mapController.areaSelectEnabled) {
        if (areaSelect.enabled && mapController.areaSelectInProgress) {
          if (map === mapController.activeMap || mapController.mapSync) {
            areaSelect._doMouseMove(ev.detail)
          }
        }
      }
    });
    useEffect(() => {
      mapController.addEventListener('mousemove', onMasterMouseMove);
      return () => {
        mapController.removeEventListener('mousemove', onMasterMouseMove);
      }
    }, [mapController, onMasterMouseMove])

    // Mouse Up
    // If area select is enabled, this the active map or map sync is on
    const onMasterMouseUp = useCallback((ev) => {
      if (mapController.areaSelectEnabled) {
        if (map === mapController.activeMap || mapController.mapSync) {
          areaSelect._doMouseUp(ev.detail)

          // Add click events to any new areas
          const areas = areaSelect.getAreas();
          for (let i = 0; i < areas.length; i++) {
            areas[i].off('click') // remove any existing handlers
            areas[i].on('click', function (e) { mapController.onAreaSelectionClicked(e) });
          }
        }
      }
    });
    useEffect(() => {
      mapController.addEventListener('mouseup', onMasterMouseUp);
      return () => {
        mapController.removeEventListener('mouseup', onMasterMouseUp);
      }
    }, [mapController, onMasterMouseUp])


    // --- MAP GRID ---

    // Grid enabled
    const onMapGridEnabled = useCallback((ev) => {
      if (graticule) {
        graticule.options.hidden = false;
        graticule.redraw();
      }
    });
    useEffect(() => {
      mapController.addEventListener('mapgridenabled', onMapGridEnabled);
      return () => {
        mapController.removeEventListener('mapgridenabled', onMapGridEnabled);
      }
    }, [mapController, onMapGridEnabled])

    // Grid disabled
    const onMapGridDisabled = useCallback((ev) => {
      if (graticule) {
        graticule.options.hidden = true;
        graticule.redraw();
      }
    });
    useEffect(() => {
      mapController.addEventListener('mapgriddisabled', onMapGridDisabled);
      return () => {
        mapController.removeEventListener('mapgriddisabled', onMapGridDisabled);
      }
    }, [mapController, onMapGridDisabled])

    return null;
  }

  // Show the current coordinates
  // Adapted from: https://react-leaflet.js.org/docs/example-external-state
  function LayerPositionInfo({ map }) {

    // State
    const [position, setPosition] = useState(map.getCenter())

    // Set map position when clicked
    const onClick = useCallback(() => {
      map.setView([0, 0], 0)
    }, [map])

    // Update displayed position when map position moves
    const onMove = useCallback(() => {
      setPosition(map.getCenter())
    }, [map]);

    // Attach listener once to map.move event.
    useEffect(() => {
      map.on('move', onMove)
      return () => {
        map.off('move', onMove)
      }
    }, [map, onMove]);

    // Display information and button
    return (
      <p className={'map-position-info'}>
        latitude: {position.lat.toFixed(4)}, longitude: {position.lng.toFixed(4)}{' '}
        <button onClick={onClick}>reset</button>
      </p>
    );
  }

  // Show statistics from the query report
  function LayerStats({ mapStatsEnabled, stats }) {

    // If not enabled, don't show
    if (!mapStatsEnabled) {
      return null;
    }

    // If null, display nothing
    if (stats === null || stats === undefined) {
      return (<p>No report.</p>);
    }

    // Display information and button
    return (
      <p className={'map-stats'}>
        Count: {stats['count'].toLocaleString()}<br />
        Sum: {stats['sum'].toLocaleString()}<br />
        Min: {stats['min'].toLocaleString()}<br />
        Max: {stats['max'].toLocaleString()}<br />
        Std: {stats['std'].toLocaleString()}<br />
        Mean: {stats['mean'].toLocaleString()}
      </p>
    );
  }

  // KDE imageOverlay - place underneath binned tile layer
  function LayerKDE({ mapController, facetRowValue, facetColValue, setKDELoading }) {

    // Refs
    const kdeRef = useRef();

    // Handle url changed
    const onUrlChanged = useCallback(() => {
      kdeRef.current.setUrl(mapController.getKdeUrl(facetRowValue, facetColValue));
      setKDELoading(true);
    }, [mapController]);

    // Attach listener once to mapController url change.
    useEffect(() => {
      mapController.addEventListener('kdeUrlChanged', onUrlChanged);
      return () => {
        mapController.removeEventListener('kdeUrlChanged', onUrlChanged);
      }
    }, [mapController, onUrlChanged])

    return (
      <Pane name='kde-pane' style={{ zIndex: 499 }}>
        <ImageOverlay
          ref={kdeRef}
          attribution=''
          url={mapController.getKdeUrl(facetRowValue, facetColValue)}
          tileSize={256}
          bounds={[[0, 0], [256, 256]]}
          opacity={1.0}
          eventHandlers={{
            load: () => {
              setKDELoading(false); // Hide spinner after loaded
            }
          }}
        />
      </Pane>
    );
  }

  // Markers layer (add/remove etc.)
  function LayerMarkers({ mapController, map }) {

    // Group of markers
    const markerGroup = useRef(L.markerClusterGroup());

    let markerIcon = L.icon({
      iconUrl: imgMarkerIcon, //'leaflet/dist/images/marker-icon.png',
      shadowUrl: imgMarkerShadow, //'leaflet/dist/images/marker-shadow.png',
      iconSize: [24, 32], // size of the icon (width, height)
      shadowSize: [48, 32], // size of the shadow (width, height)
      iconAnchor: [12, 32], // point of the icon which will correspond to marker's location. Assumes tip is center (lng, lat)
      shadowAnchor: [12, 32],  // the same for the shadow
      popupAnchor: [0, -20] // point from which the popup should open relative to the iconAnchor
    })

    // Handle markers changed
    const onMarkersChanged = useCallback((ev) => {

      // Get the items
      const markerItems = ev.detail;

      // Remove any existing marker layers
      /*
      map.eachLayer((layer) => {
        if (layer instanceof L.MarkerClusterGroup)
        {
          map.removeLayer(layer)
        }
      })
      */

      // Create cluster of markers
      //markerGroup.current.clearLayers();

      for (let i = 0; i < markerItems.length; i++) {
        const markerItem = markerItems[i]
        const lat = markerItem['y'];
        const lng = markerItem['x'];
        const seq = markerItem['seq'].toString();
        let marker = L.marker([lat, lng], { icon: markerIcon });
        marker.bindTooltip((i+1).toString(),
          {
            permanent: true,
            direction: 'top',
            className: 'markerLabel'
          }
        );
        marker.addTo(map);
        marker.bindPopup(seq);
        //console.log('Adding marker to layer')
        //console.log(marker)
        //markerGroup.current.addLayer(marker)
      }

      // Add layer to map
      //map.addLayer(markerGroup.current);

      // Add to the map
      //map.addLayer(markerGroup.current);
    }, [mapController]);


    // Attach listener once to mapController markers changed.
    useEffect(() => {
      mapController.addEventListener('markerschanged', onMarkersChanged);
      return () => {
        mapController.removeEventListener('markerschanged', onMarkersChanged);
      }
    }, [mapController, onMarkersChanged])


    return null;

  }

  // Binned Tile layer
  // See GridLayerOptions: https://leafletjs.com/reference-1.1.0.html#gridlayer-tilesize */}
  function LayerBinnedTile({ mapController, facetRowValue, facetColValue, setBinnedLoading }) {

    // Refs
    const tileRef = useRef();

    // Handle url changed
    const onUrlChanged = useCallback(() => {
      tileRef.current.setUrl(mapController.getTileUrl(facetRowValue, facetColValue));
    }, [mapController]);

    // Attach listener once to mapController url change.
    useEffect(() => {
      mapController.addEventListener('tileUrlChanged', onUrlChanged);
      return () => {
        mapController.removeEventListener('tileUrlChanged', onUrlChanged);
      }
    }, [mapController, onUrlChanged])

    // Render
    return (
      <Pane name='tile-pane' style={{ zIndex: 505 }}>
        <TileLayer
          ref={tileRef}
          attribution=''
          url={mapController.getTileUrl(facetRowValue, facetColValue)}
          tileSize={256}
          bounds={[[0, 0], [256, 256]]}
          updateWhenIdle={true} // don't update while panning
          updateWhenZooming={false} // only update once finished zooming
          eventHandlers={{
            load: () => {
              setBinnedLoading(false); // Hide spinner after loaded
            },
            loading: () => {
              setBinnedLoading(true); // When loading
            }
          }}
        />
      </Pane>
    );
  }

  // Grid of coordinates (use panes to keep grid lines in front of image layers.
  function LayerGridCoordinates({ setGraticule }) {

    // Refs
    const map = useMap();

    // Set up
    useEffect(() => {
      var options = {
        hidden: true,
        interval: 20,
        showOriginLabel: true,
        redraw: 'move',
        zoomIntervals: [
          { start: 0, end: 3, interval: 50 },
          { start: 4, end: 5, interval: 5 },
          { start: 6, end: 20, interval: 1 }
        ]
      };
      const graticule = L.simpleGraticule(options)

      // Set line style
      //graticule.lineStyle.color = '#BBBBBB'; //'#111';
      graticule.lineStyle.color = '#888888'; 
      graticule.lineStyle.opacity = 0.8;
      graticule.lineStyle.weight = 0.2;

      // Add it to the map
      graticule.addTo(map);

      // Store it
      setGraticule(graticule);
    }, [])

    return (
      <Pane name='simple-graticule-pane' style={{ zIndex: 510 }} />
    );
  }

  // Area Selection target pane (use panes to keep selection polyline in front of image layers.
  // Area selection is handled by MapEventsController
  function LayerAreaSelect() {
    return (
      <Pane name='area-select-pane' style={{ zIndex: 520 }} />
    );
  }

  // Anchor component
  function LayerMapAnchors({ map, anchors }) {

    for (let k in anchors) {
      let anchor = anchors[k];
      let x = anchor['x'];
      let y = anchor['y'];
      let class_rgb = anchor['class_rgb'];
      console.log(class_rgb[0]);
      let anchor_name = anchor['anchor_name'];

      let circle = L.circle([y, x], { // NOTE! Switch around, as Leaflet is lat/long (y,x)
        color: 'rgb(' + class_rgb[0] * 255 + ',' +
          class_rgb[1] * 255 + ',' +
          class_rgb[2] * 255 + ')',

        fillColor: '#f03',
        fillOpacity: 0.5,
        radius: 0.1
      }).bindTooltip(anchor_name)
        .bindPopup(anchor_name)
        .addTo(map);
    }
    return null
  }

  return (
    <DisplayMap
      id={id}
      mapController={mapController}
      mapTheme={mapTheme}
      facetRowValue={facetRowValue}
      facetColValue={facetColValue}
      mapLabel={mapLabel}
      mapStatsEnabled={mapStatsEnabled}
      stats={stats}
    />
  );
}

export default LeafletItem;