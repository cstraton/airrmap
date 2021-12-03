import React, { useState, useRef, useCallback, useEffect } from "react"
import 'fomantic-ui-css/semantic.css';
import './index.css';

import {
  Button, Container, Icon, Menu, Modal,
  Popup, Segment, Sidebar, Tab
} from 'semantic-ui-react';

import usePersistedState from "./assets/components/persisted_state"
import LeafletGrid from './assets/components/leaflet_grid.js'
import FilterSelection2 from './assets/components/filter_selection2.js'
import KDESettings from './assets/components/kde_settings.js'
import ROIReport from './assets/components/roi_report.js'
import SeqLocator from './assets/components/seq_locator.js'
import MasterMapController from './assets/js/master_map_controller.js'
import CONFIG from '../config.json';

function App(props) {

  // State and vars
  // NOTE!: If changing, consider updating 'For loaded properties (using usePersistedState):' (below)
  //        e.g. mapController.setBinnedEnabled(binnedEnabled).
  //        Also: (1) add property to master_map_controller.js.
  //              (2) add callback (e.g. setMyProperty, below)
  const [sidebarVisible, setSidebarVisible] = useState(true);
  const [filters, _setFilters] = useState({});  // User filters

  // Binned tile settings
  const [binnedEnabled, _setBinnedEnabled] = usePersistedState('store-app-binnedEnabled', false);
  const [brightness, _setBrightness] = usePersistedState('store-app-brightness', 1.0); // Map brightness multiplier slider (tiles/binned)
  const [numBins, _setNumBins] = usePersistedState('store-app-numBins', 1); // 0 = 256, 1 = 128, 2 = 64 etc.

  // KDE settings
  const [kdeEnabled, _setKdeEnabled] = usePersistedState('store-app-kdeEnabled', true);
  const [kdeBandwidth, _setKdeBandwidth] = usePersistedState('store-app-kdeBandwidth', 0.8); // KDE kernel bandwidth
  const [kdeColormap, _setKdeColormap] = usePersistedState('store-app-kdeColormap', 'RdBu'); // Colour map to use
  const [kdeColormapInvert, _setKdeColormapInvert] = usePersistedState('store-app-kdeColormapInvert', true); // Invert colours
  const [kdeBrightness, _setKdeBrightness] = usePersistedState('store-app-kdeBrightness', 2.0); // KDE brightness multiplier slider
  const [kdeRelativeMode, _setKdeRelativeMode] = usePersistedState('store-app-kdeRelativeMode', 'SINGLE');
  const [kdeRelativeSelection, _setKdeRelativeSelection] = usePersistedState('store-app-kdeRelativeSelection', 'ALL');
  const [kdeRowName, _setKdeRowName] = usePersistedState('store-app-kdeRowName', '');
  const [kdeColumnName, _setKdeColumnName] = usePersistedState('store-app-kdeColumnName', '');

  // Query report information
  const [queryReport, setQueryReport] = useState({});
  const [facetRowValues, setFacetRowValues] = useState(['']);
  const [facetColValues, setFacetColValues] = useState(['']);
  const [value1Min, _setValue1Min] = useState(0);
  const [value1Max, _setValue1Max] = useState(88); // Arbitrary, brightness scaling.

  // ROI reports and selection
  const [selectEnabled, _setSelectEnabled] = useState(false) // Whether area selection (lasso) is currently enabled.
  const [roiReport, setROIReport] = useState(null); // region report
  const [roiReportOpen, setROIReportOpen] = useState(false); // Whether ROI report modal is shown
  const [roiFacetRowValues, setROIFacetRowValues] = useState([]); // The facet row values that the roi report relates to.
  const [roiFacetColValues, setROIFacetColValues] = useState([]); // The facet col values that the roi report relates to.

  // Markers
  const [markerItems, _setMarkerItems] = useState([]); // Map marker items

  // Status message
  const [appStatusMessage, setAppStatusMessage] = useState('Welcome to AIRR Map.'); // Status message
  const [appStatusHidden, setAppStatusHidden] = useState(false); // True/False
  const [appStatusType, setAppStatusType] = useState('warning'); // 'none', 'info', 'warning', 'error', 'positive'
  const [appStatusLoading, setAppStatusLoading] = useState(false); // true or false

  // Sync maps and map control
  const [mapSync, _setMapSync] = useState(false) // Whether maps are synced or not
  const [mapGridEnabled, _setMapGridEnabled] = useState(false) // 
  const [mapStatsEnabled, _setMapStatsEnabled] = usePersistedState('store-app-mapStatsEnabled', false);
  const [mapController, setMapController] = useState(new MasterMapController());


  // For loaded properties (using usePersistedState):
  // -------------------------------------------------
  // Set loaded mapcontroller properties
  // If not done, loaded properties from usePersistedState
  // won't take effect until changing the value / triggering
  // the onChange event in the UI.
  mapController.setBinnedEnabled(binnedEnabled)
  mapController.setNumBins(numBins)
  mapController.setBrightness(brightness)
  mapController.setKdeEnabled(kdeEnabled)
  mapController.setKdeBandwidth(kdeBandwidth)
  mapController.setKdeBrightness(kdeBrightness)
  mapController.setKdeColormap(kdeColormap)
  mapController.setKdeColormapInvert(kdeColormapInvert)
  mapController.setKdeRelativeMode(kdeRelativeMode)
  mapController.setKdeRelativeSelection(kdeRelativeSelection)
  mapController.setKdeRowName(kdeRowName)
  mapController.setKdeColumnName(kdeColumnName)
  mapController.setMapStatsEnabled(mapStatsEnabled)

  // --- Callbacks ---

  const setFilters = (filters) => {
    mapController.filters = filters;
    _setFilters(filters);
  }

  const setBinnedEnabled = (value) => {
    mapController.setBinnedEnabled(value);
    _setBinnedEnabled(value);
  }

  const setNumBins = (numBins) => {
    mapController.setNumBins(numBins);
    _setNumBins(numBins);
  }

  const setBrightness = (brightness) => {
    mapController.setBrightness(brightness);
    _setBrightness(brightness);
  }

  const setKdeEnabled = (value) => {
    mapController.setKdeEnabled(value);
    _setKdeEnabled(value);
  }

  const setKdeBrightness = (value) => {
    mapController.setKdeBrightness(value);
    _setKdeBrightness(value);
  }

  const setKdeBandwidth = (kdeBandwidth) => {
    mapController.setKdeBandwidth(kdeBandwidth);
    _setKdeBandwidth(kdeBandwidth);
  }

  const setKdeColormap = (value) => {
    mapController.setKdeColormap(value);
    _setKdeColormap(value);
  }

  const setKdeColormapInvert = (value) => {
    mapController.setKdeColormapInvert(value);
    _setKdeColormapInvert(value);
  }

  const setKdeRelativeMode = (value) => {
    mapController.setKdeRelativeMode(value);
    _setKdeRelativeMode(value);
  }

  const setKdeRelativeSelection = (value) => {
    mapController.setKdeRelativeSelection(value);
    _setKdeRelativeSelection(value);
  }

  const setKdeRowName = (value) => {
    mapController.setKdeRowName(value);
    _setKdeRowName(value);
  }

  const setKdeColumnName = (value) => {
    mapController.setKdeColumnName(value);
    _setKdeColumnName(value);
  }

  const setValue1Min = (value1Min) => {
    mapController.setValue1Min(value1Min);
    _setValue1Min(value1Min);
  }

  const setValue1Max = (value1Max) => {
    mapController.setValue1Max(value1Max);
    _setValue1Max(value1Max);
  }

  const setMapSync = (mapSync) => {
    mapController.setMapSync(mapSync);
    _setMapSync(mapSync);
  }

  const setMapStatsEnabled = (value) => {
    mapController.setMapStatsEnabled(value);
    _setMapStatsEnabled(value);
  }

  const setMapGridEnabled = (gridEnabled) => {
    mapController.setMapGridEnabled(gridEnabled)
    _setMapGridEnabled(gridEnabled);
  }

  const setMarkerItems = (markerItems) => {
    mapController.setMarkerItems(markerItems)
    _setMarkerItems(markerItems);
  }

  const setSelectEnabled = (selectEnabled) => {
    mapController.setAreaSelect(selectEnabled);
    _setSelectEnabled(selectEnabled);
  }

  const setAppStatus = useCallback((message, statusType, isLoading) => {
    // Update the app status message
    setAppStatusMessage(message);
    setAppStatusType(statusType);
    setAppStatusLoading(isLoading);
    setAppStatusHidden(!(message == '' || message != null))
  }, []);


  const submitFiltersHandler = useCallback((data) => {
    // Handle the 'Submit' button on the Filters pane.

    //console.log(data);

    // Clear previous report
    setROIReport(null);

    // Don't have maps synced initially
    setMapSync(false);

    // Request query on server
    runQuery(data);

  }, []);


  // --- Init ---

  // MapController, area selection event 
  const onAreaSelectionClicked = useCallback((e) => {

    // Turn off selection to stop drawing on mousemove
    setSelectEnabled(false);

    // Request the ROI report
    requestROIReport(e.detail)

  })

  useEffect(() => {
    mapController.addEventListener('areaselectionclicked', onAreaSelectionClicked);
    return () => {
      mapController.removeEventListener('areaselectionclicked', onAreaSelectionClicked);
    }
  }, [mapController, onAreaSelectionClicked])


  function runQuery(query) {
    // Prepare the data on the server
    // by running a query based on the
    // selected filters.

    // This will cache the results,
    // ready for tile rendering.
    const data_json_uri = encodeURIComponent(JSON.stringify(query));

    // Show loading
    setAppStatus('Running query, please wait...', 'info', true)

    fetch(CONFIG.baseUrl + 'tiles/data?q=' + data_json_uri, {
      method: 'GET',
      mode: 'cors',
      headers: { "Content-type": "application/json; charset=UTF-8" }
    })
      .then(response => response.json())
      .then(report => {
        //console.log(query);

        // Format status message to show user
        let status_text = report['status_message'] + ' ' +
          report['record_count'].toLocaleString() + ' records in ' +
          report['query_time'].toLocaleString() + 's' +
          (report['cached'] ? ' (cached)' : '')

        let message_type = report['success'] ? 'positive' : 'error'
        setAppStatus(status_text, message_type, false)

        // Set state - will be forwarded to maps to start rendering
        setFilters(query);
        setQueryReport(report);
        setFacetRowValues(report['facet_row_values']);
        setFacetColValues(report['facet_col_values']);
        setValue1Min(report['value1_min']);
        setValue1Max(report['value1_max']);

        // Sync on by default
        setMapSync(true);
      })
      .catch(err => {
        //console.log(data);
        setAppStatus('Error! ' + err.toString(), 'error', false)
      })
  }

  function requestSeqCoordinates(env_name, seqs) {
    // Submit a request to compute the coordinates for the selected sequences.

    // Build the request
    const _request = {
      env_name: env_name,
      seqs: seqs
    }

    // Loading...
    setAppStatus('Loading sequence coordinates...', 'info', true);

    // Compute the coordinates
    fetch(CONFIG.baseUrl + 'items/seqcoords/', {
      method: 'POST',
      body: JSON.stringify(_request),
      mode: 'cors',
      headers: { "Content-type": "application/json; charset=UTF-8" }
    })
      .then(response => response.json())
      //.then(data => console.log(data))
      .then(data => {
        setAppStatus('Completed', 'none', false);
        console.log(data);
        setMarkerItems(data);
      })
      .catch(err => {
        setAppStatus(err.toString(), 'error', false);
      });
  }

  function requestROIReport(e) {
    // Submit a ROI request, with polygon selection

    // Get the points of clicked polygon
    const latlngs = e.target.getLatLngs()[0];
    const bounds = e.target.getBounds();

    //console.log('Latlngs length: ' + latlngs.length.toString());

    // Custom attributes passed to MapContainer (in leaflet_item.js) are available in .options
    // See: https://stackoverflow.com/questions/59149848/how-to-pass-and-get-additional-data-values-from-react-leaflet-marker
    const facetRowValue = mapSync ? '' : e.target._map.options.smFacetRowValue // If sync on, get all facets ('')
    const facetColValue = mapSync ? '' : e.target._map.options.smFacetColValue // If sync on, get all facets ('')

    // Create request
    let _request = {
      facetRowValue: facetRowValue,
      facetColValue: facetColValue,
      filters: filters,
      latlngs: latlngs,
      bounds: bounds
    }

    // Loading...
    setAppStatus('Loading report, please wait...', 'info', true);

    // Load ROI summary
    fetch(CONFIG.baseUrl + 'items/polyroi/', {
      method: 'POST',
      body: JSON.stringify(_request),
      mode: 'cors',
      headers: { "Content-type": "application/json; charset=UTF-8" }
    })
      .then(response => response.json())
      //.then(data => console.log(data))
      .then(data => {
        setAppStatus('Completed', 'none', false);
        setROIFacetRowValues(mapSync ? facetRowValues : [facetRowValue]); // If sync was on, set to all, otherwise the selected row.
        setROIFacetColValues(mapSync ? facetColValues : [facetColValue]); // If sync was on, set to all, otherwise the selected col.
        setROIReport(data)
        setROIReportOpen(true)
      })
      .catch(err => {
        setAppStatus(err.toString(), 'error', false);
        //console.log(err)
      });
  }

  function RenderROIReport() {
    return (
      <Modal
        className={'modal-roi-report'}
        closeIcon={true}
        dimmer={'blurring'}
        onClose={() => setROIReportOpen(false)}
        onOpen={() => setROIReportOpen(true)}
        open={roiReportOpen}
        size={'fullscreen'}  // fullscreen or large
      >
        <Modal.Header>Results</Modal.Header>
        <Modal.Content scrolling={true}>
          <ROIReport
            //If updating this, also update ROIReport use above in menuItem
            report={roiReport}
            facetRowValues={roiFacetRowValues}
            facetColValues={roiFacetColValues}
            setAppStatus={setAppStatus}
          />
        </Modal.Content>
      </Modal>
    );
  }

  // Construct tab panes
  // Use `render: () =>` instead of `pane:` for <Tab> property renderActiveOnly={true}.
  // If using renderActiveOnly={false}, react-semantic-ui-range slider doesn't render
  // <value> correctly (issue with component / race condition with rendering?).
  function RenderPanes() {
    return (
      [
        {
          menuItem: { key: 'filters1', content: 'Data' },
          render: () =>
            <Tab.Pane key='filters-pane' className={'no-border'}>
              <FilterSelection2
                submitHandler={submitFiltersHandler}
                appStatusLoading={appStatusLoading}
              />
            </Tab.Pane>
        },
        {
          // KDE config
          menuItem: { key: 'kde', content: 'Rendering' },
          render: () =>
            <Tab.Pane key='kde-pane' className={'no-border'}>
              <KDESettings
                binnedEnabled={binnedEnabled}
                setBinnedEnabled={setBinnedEnabled}
                brightness={brightness}
                setBrightness={setBrightness}
                numBins={numBins}
                setNumBins={setNumBins}
                kdeEnabled={kdeEnabled}
                setKdeEnabled={setKdeEnabled}
                kdeBandwidth={kdeBandwidth}
                setKdeBandwidth={setKdeBandwidth}
                kdeBrightness={kdeBrightness}
                setKdeBrightness={setKdeBrightness}
                kdeColormap={kdeColormap}
                setKdeColormap={setKdeColormap}
                kdeColormapInvert={kdeColormapInvert}
                setKdeColormapInvert={setKdeColormapInvert}
                kdeRelativeMode={kdeRelativeMode}
                setKdeRelativeMode={setKdeRelativeMode}
                kdeRelativeSelection={kdeRelativeSelection}
                setKdeRelativeSelection={setKdeRelativeSelection}
                kdeRowName={kdeRowName}
                setKdeRowName={setKdeRowName}
                kdeColumnName={kdeColumnName}
                setKdeColumnName={setKdeColumnName}
                facetRowValues={facetRowValues}
                facetColValues={facetColValues}
              />
            </Tab.Pane>

        },
        {
          // ROI Selection
          menuItem: { key: 'roi-report', content: 'Selection' },
          render: () =>
            <Tab.Pane key='roi-report-pane' className={'no-border'}>
              <ROIReport
                //If updating this, also update RenderROIReport below
                report={roiReport}
                facetRowValues={facetRowValues}
                facetColValues={facetColValues}
                setAppStatus={setAppStatus}
              />
            </Tab.Pane>
        },
        {
          // Sequence locator list
          menuItem: { key: 'seq-locator', content: 'Markers' },
          render: () =>
            <Tab.Pane key='seq-locator-pane' className={'no-border'}>
              <SeqLocator
                env_name={filters['env_name']}
                setAppStatus={setAppStatus}
                submitCallback={requestSeqCoordinates}
              />
            </Tab.Pane>
        }
      ]
    );
  }

  function RenderStatus({ appStatusLoading, appStatusType, appStatusMessage }) {
    return (
      <Popup
        content={appStatusMessage}
        mouseEnterDelay={CONFIG.tooltips.mouseEnterDelay}
        mouseLeaveDelay={CONFIG.tooltips.mouseLeaveDelay}
        trigger={
          <span className='toolbar-status-inner'>
            {appStatusMessage}
          </span>
        }
      />
    );
  }

  function RenderTopHeader() {
    return (
      //Top Menu 
      <div className={'toolbar-top-outer'}>
        <div className={'toolbar-top-inner'}>
          <Menu className={'no-margin'} size={'tiny'} color={'green'} compact>

            {/* Sidebar visibility */}
            <Menu.Item
              name='Show Sidebar'
              toggle
              active={sidebarVisible}
              onClick={() => setSidebarVisible(!(sidebarVisible))}
            >
              <Popup
                content={CONFIG.tooltips.toolbar.sidebar}
                mouseEnterDelay={CONFIG.tooltips.mouseEnterDelay}
                mouseLeaveDelay={CONFIG.tooltips.mouseLeaveDelay}
                trigger={
                  <Icon className='bi bi-list' />
                }
              />
            </Menu.Item>

            {/* App name */}
            <Menu.Item
              name='AIRR Map'
            />

            {/* Sync button */}
            <Menu.Item>
              <Popup
                content={CONFIG.tooltips.toolbar.sync}
                mouseEnterDelay={CONFIG.tooltips.mouseEnterDelay}
                mouseLeaveDelay={CONFIG.tooltips.mouseLeaveDelay}
                trigger={
                  <Button icon labelPosition='left' toggle active={mapSync} onClick={() => setMapSync(!(mapSync))}>
                    <Icon className='bi bi-arrow-left-right' />
                    Sync
                  </Button>
                }
              />
            </Menu.Item>

            {/* Grid button */}
            <Menu.Item>
              <Popup
                content={CONFIG.tooltips.toolbar.grid}
                mouseEnterDelay={CONFIG.tooltips.mouseEnterDelay}
                mouseLeaveDelay={CONFIG.tooltips.mouseLeaveDelay}
                trigger={
                  <Button icon labelPosition='left' toggle active={mapGridEnabled} onClick={() => setMapGridEnabled(!(mapGridEnabled))}>
                    <Icon className='bi bi-grid-3x3' />
                    Grid
                  </Button>
                }
              />
            </Menu.Item>

            {/* Stats button */}
            <Menu.Item>
              <Popup
                content={CONFIG.tooltips.toolbar.stats}
                mouseEnterDelay={CONFIG.tooltips.mouseEnterDelay}
                mouseLeaveDelay={CONFIG.tooltips.mouseLeaveDelay}
                trigger={
                  <Button icon labelPosition='left' toggle active={mapStatsEnabled} onClick={() => setMapStatsEnabled(!(mapStatsEnabled))}>
                    <Icon className='bi bi-text-indent-left' />
                    Stats
                  </Button>
                }
              />
            </Menu.Item>

            {/* Area select toggle */}
            <Menu.Item
              name='Enable area select'
              active={selectEnabled}
              onClick={() => setSelectEnabled(!(selectEnabled))}
            >
              <Popup
                content={CONFIG.tooltips.toolbar.selectDraw}
                mouseEnterDelay={CONFIG.tooltips.mouseEnterDelay}
                mouseLeaveDelay={CONFIG.tooltips.mouseLeaveDelay}
                trigger={
                  <Icon className='bi bi-pen' />
                }
              />
            </Menu.Item>

            {/* Remove last selection */}
            <Menu.Item
              name='Remove last selection'
              onClick={() => mapController.onAreaSelectRemoveLast()}
            >
              <Popup
                content={CONFIG.tooltips.toolbar.selectUndo}
                mouseEnterDelay={CONFIG.tooltips.mouseEnterDelay}
                mouseLeaveDelay={CONFIG.tooltips.mouseLeaveDelay}
                trigger={
                  <Icon className='bi bi-arrow-counterclockwise' />
                }
              />
            </Menu.Item>

            {/* Remove all selections */}
            <Menu.Item
              name='Remove all selections'
              onClick={() => mapController.onAreaSelectClear()}
            >
              <Popup
                content={CONFIG.tooltips.toolbar.selectRemoveAll}
                mouseEnterDelay={CONFIG.tooltips.mouseEnterDelay}
                mouseLeaveDelay={CONFIG.tooltips.mouseLeaveDelay}
                trigger={
                  <Icon className='bi bi-x' />
                }
              />
            </Menu.Item>

            {/* Show report */}
            <Menu.Item
              name='Show report'
              active={roiReportOpen}
              onClick={() => setROIReportOpen(!(roiReportOpen))}
            >
              <Popup
                content={CONFIG.tooltips.toolbar.report}
                mouseEnterDelay={CONFIG.tooltips.mouseEnterDelay}
                mouseLeaveDelay={CONFIG.tooltips.mouseLeaveDelay}
                trigger={
                  <Icon className='bi bi-bar-chart-fill' />
                }
              />
            </Menu.Item>
            <Menu.Item>
              {/* Status */}
              <RenderStatus
                appStatusLoading={appStatusLoading}
                appStatusType={appStatusType}
                appStatusMessage={appStatusMessage}
              />
            </Menu.Item>
          </Menu >
        </div>
      </div>
    );
  }

  // Render the layout function
  function renderLayout() {
    return (

      <Container fluid={true} className='container-fixed-height-100'>

        {/* Full screen modal report */}
        <RenderROIReport id='modal-fullscreen-roi-report' />

        {/* The top header */}
        <RenderTopHeader />

        {/* Left Sidebar */}
        {/* Adapted from: https://react.semantic-ui.com/modules/sidebar/#states-visible */}
        <Sidebar.Pushable className={'no-margin no-border'} as={Segment}>
          <Sidebar
            as={Segment}
            animation={'scale down'}
            visible={sidebarVisible}
            width={'very wide'}
          >
            {/*Tab Container */}
            <Tab menu={{ secondary: true, pointing: true, color: 'blue' }} panes={RenderPanes()} renderActiveOnly={true} />
          </Sidebar>

          {/* Right Map Grid */}
          <Sidebar.Pusher>
            <LeafletGrid
              key='leaflet-grid-main'
              mapController={mapController}
              facetRowValues={facetRowValues}
              facetColValues={facetColValues}
              queryReport={queryReport}
            />
          </Sidebar.Pusher>
        </Sidebar.Pushable>
      </Container>
    );
  }

  // Render layout
  return (
    renderLayout()
  );
}

export default App;