// Central controller for Leaflet maps.
// Enables syncing and event coordination.
// Should handle any property relating to Leaflet map instances
// or any property that should be passed in tile/image layer URLs
import CONFIG from '../../../config.json';

export default class MasterMapController extends EventTarget {

  // Example:
  //masterMapController.addEventListener('tileUrlChanged', (e) => { console.log('URL Changed', e.detail) });
  //masterMapController.setKdeBandwidth(2.0);

  constructor() {
    super(); // Need 'this'.

    // Use setter() functions below, don't set directly.
    this.position = [0.0, 0.0];
    this.zoom = 0;
    this.activeMap = null;
    this.mapSync = false;
    this.mapGridEnabled = false;
    this.areaSelectEnabled = false;
    this.areaSelectInProgress = false;
    this.kdeBrightness = 1.0; // KDE brightness
    this.kdeBandwidth = 1.0;
    this.kdeColormap = 'RdBu';
    this.kdeColormapInvert = false;
    this.kdeRelativeMode = 'ALL';
    this.kdeRelativeSelection = 'ALL';
    this.kdeRowName = '';
    this.kdeColumnName = '';
    this.brightness = 1.0; // Map brightness multiplier slider (tile/binned)
    this.numBins = 2;  // 0 = 256, 1 = 128, 2 = 64 etc.
    this.filters = {};
    this.markerItems = []; // Map marker items.
    this.value1Min = 0.0;
    this.value1Max = 1.0;
    this.tileUrlTimer = null; // timer
    this.KdeUrlTimer = null; // timer
    this.timerWaitms = 1500;
    this.baseUrl = CONFIG.baseUrl;
  }

  onTileUrlChanged() {

    // Cancel previous timer
    if (this.tileUrlTimer) {
      clearTimeout(this.tileUrlTimer);
    }

    // Limit the number of requests to API
    this.tileUrlTimer = setTimeout(() => {
      //const evt = new CustomEvent('urlChanged', { detail: { tileUrl, kdeUrl} });
      const evt = new CustomEvent('tileUrlChanged', {});
      this.dispatchEvent(evt);
    }, this.timerWaitms);
  }


  onKdeUrlChanged() {

    // Cancel previous timer
    if (this.KdeUrlTimer) {
      clearTimeout(this.KdeUrlTimer);
    }

    // Limit the number of requests to API
    this.KdeUrlTimer = setTimeout(() => {
      //const evt = new CustomEvent('urlChanged', { detail: { tileUrl, kdeUrl} });
      const evt = new CustomEvent('kdeUrlChanged', {});
      this.dispatchEvent(evt);
    }, this.timerWaitms);
  }

  // When the map moves (not the mouse)
  onMove(ev) {
    if (this.mapSync) {
      const evt = new CustomEvent('moved', {});
      this.dispatchEvent(evt);
    }
  }

  // Map grid enabled
  onMapGridEnabled() {
    const evt = new CustomEvent('mapgridenabled', {});
    this.dispatchEvent(evt);
  }

  // Map grid disabled
  onMapGridDisabled() {
    const evt = new CustomEvent('mapgriddisabled', {});
    this.dispatchEvent(evt);
  }

  onMarkersChanged() {
    const evt = new CustomEvent('markerschanged', {'detail': this.markerItems});
    this.dispatchEvent(evt);
  }

  onMouseDown(mapevt) {
    // Controls drawing for area selection
    if (this.areaSelectEnabled) {
      this.areaSelectInProgress = true;
    }

    // Raise the event
    const evt = new CustomEvent('mousedown', { detail: mapevt });
    this.dispatchEvent(evt);
  }

  onMouseUp(mapevt) {
    this.areaSelectInProgress = false;
    const evt = new CustomEvent('mouseup', { detail: mapevt });
    this.dispatchEvent(evt);
  }

  onMouseMove(mapevt) {
    const evt = new CustomEvent('mousemove', { detail: mapevt });
    this.dispatchEvent(evt);
  }

  onAreaSelectEnabled() {
    const evt = new CustomEvent('areaselectenabled', {});
    this.dispatchEvent(evt);
  }

  onAreaSelectDisabled() {
    this.areaSelectInProgress = false;
    const evt = new CustomEvent('areaselectdisabled', {});
    this.dispatchEvent(evt);
  }

  onAreaSelectClear() {
    this.areaSelectInProgress = false;
    const evt = new CustomEvent('areaselectclear', {});
    this.dispatchEvent(evt);
  }

  onAreaSelectRemoveLast() {
    this.areaSelectInProgress = false;
    const evt = new CustomEvent('areaselectremovelast', {});
    this.dispatchEvent(evt);
  }

  onAreaSelectionClicked(e) {
    // When area selection is clicked (e.g. request roi report)
    const evt = new CustomEvent('areaselectionclicked', { 'detail': e });
    this.dispatchEvent(evt);
  }

  setKdeBrightness(value) {
    this.kdeBrightness = value;
    this.onKdeUrlChanged();
  }

  setKdeBandwidth(value) {
    this.kdeBandwidth = value;
    this.onKdeUrlChanged();
  }

  setKdeColormap(value) {
    this.kdeColormap = value;
    this.onKdeUrlChanged();
  }

  setKdeColormapInvert(value) {
    this.kdeColormapInvert=value;
    this.onKdeUrlChanged();
  }

  setKdeRelativeMode(value) {
    this.kdeRelativeMode = value;
    this.onKdeUrlChanged();
  }

  setKdeRelativeSelection(value) {
    this.kdeRelativeSelection = value;
    this.onKdeUrlChanged();
  }

  setKdeRowName(value) {
    this.kdeRowName = value;
    this.onKdeUrlChanged();
  }

  setKdeColumnName(value) {
    this.kdeColumnName = value;
    this.onKdeUrlChanged();
  }

  setNumBins(value) {
    this.numBins = value;
    this.onTileUrlChanged();
  }

  setBrightness(value) {
    this.brightness = value;
    this.onTileUrlChanged();
  }

  setValue1Min(value) {
    this.value1Min = value;
    this.onTileUrlChanged();
  }

  setValue1Max(value) {
    this.value1Max = value;
    this.onTileUrlChanged();
  }

  setActiveMap(value) {
    if (value === this.activeMap) {
      console.log('mapController: activemap already set.');
    } else {
      console.log('mapController: activemap set.');
      this.activeMap = value;
    }
  }

  setMapSync(value) {
    this.mapSync = value;

    // Trigger move update
    if (this.mapSync) {
      this.onMove()
    }
  }

  setMapGridEnabled(value) {
    this.mapGridEnabled = value;
    if (this.mapGridEnabled) {
      this.onMapGridEnabled();
    } else {
      this.onMapGridDisabled();
    }
  }

  setMarkerItems(value) {
    this.markerItems = value;
    this.onMarkersChanged()
  }

  setAreaSelect(value) {
    if (value !== this.areaSelectEnabled) {
      this.areaSelectEnabled = value;
      if (this.areaSelectEnabled) {
        this.onAreaSelectEnabled()
      } else {
        this.onAreaSelectDisabled()
      }
    }

  }

  getTileUrl(facetRowValue, facetColValue) {
    return this.baseUrl + "tiles/imagery/binned/{z}/{x}/{y}.png?a=1" +
      "&brightness=" + encodeURIComponent(this.brightness) +
      "&bins=" + encodeURIComponent(this.numBins) +
      "&q=" + encodeURIComponent(JSON.stringify(this.filters)) +
      "&v1min=" + encodeURIComponent(JSON.stringify(this.value1Min)) +
      "&v1max=" + encodeURIComponent(JSON.stringify(this.value1Max)) +
      "&fr=" + encodeURIComponent(facetRowValue) +
      "&fc=" + encodeURIComponent(facetColValue)
  }

  getKdeUrl(facetRowValue, facetColValue) {
    // -1 y for L.CRS coordinate system. Currently global (0,0,-1)
    return this.baseUrl + "tiles/imagery/kde-diff/0/0/-1.png?a=1" +
      "&kdebrightness=" + encodeURIComponent(this.kdeBrightness) + 
      "&kdebw=" + encodeURIComponent(this.kdeBandwidth) +
      "&kdecm=" + encodeURIComponent(this.kdeColormap) + 
      "&kdecminv=" + encodeURIComponent(this.kdeColormapInvert) +
      "&kderm=" + encodeURIComponent(this.kdeRelativeMode) + 
      "&kdesm=" + encodeURIComponent(this.kdeRelativeSelection) + 
      "&kderowname=" + encodeURIComponent(this.kdeRowName) + 
      "&kdecolname=" + encodeURIComponent(this.kdeColumnName) + 
      "&q=" + encodeURIComponent(JSON.stringify(this.filters)) +
      "&v1min=" + encodeURIComponent(JSON.stringify(this.value1Min)) +
      "&v1max=" + encodeURIComponent(JSON.stringify(this.value1Max)) +
      "&fr=" + encodeURIComponent(facetRowValue) +
      "&fc=" + encodeURIComponent(facetColValue);
  }

}
