
// KDE settings pane (dropdowns and sliders)

import React, { useEffect, useState } from 'react'
import { Checkbox, Divider, Dropdown, Form, Grid, Header, Icon, Label } from 'semantic-ui-react'
import { Slider } from 'react-semantic-ui-range'
import usePersistedState from "../../assets/components/persisted_state"
import 'bootstrap-icons/font/bootstrap-icons.css';
import './css/filter_selection.css';
import '../../index.css';
import CONFIG from '../../../config.json';


function KDESettings({
  brightness, setBrightness,
  numBins, setNumBins,
  kdeBrightness, setKdeBrightness,
  kdeBandwidth, setKdeBandwidth,
  kdeColormap, setKdeColormap,
  kdeColormapInvert, setKdeColormapInvert,
  kdeRelativeMode, setKdeRelativeMode,
  kdeRelativeSelection, setKdeRelativeSelection,
  kdeRowName, setKdeRowName,
  kdeColumnName, setKdeColumnName,
  facetRowValues, facetColValues }) {

  // State
  const [kdeOptions, setKdeOptions] = usePersistedState(null);

  // Num Bins Slider
  const numBinsSlider = {
    //start: 0,
    min: 0,
    max: 4,
    step: 1,
    onChange: value => {
      setNumBins(value);
    }
  }

  // Brightness slider settings
  const brightnessSlider = {
    //start: 0.0,
    min: 0.0,
    max: 200,
    step: 2.0,
    onChange: value => {
      setBrightness(value);
    }
  };

  // KDE bandwidth slider settings
  const kdeBandwidthSliderSettings = {
    //start: 0.0,
    min: 0.0,
    max: 2.0,
    step: 0.1,
    onChange: value => {
      setKdeBandwidth(value);
    }
  };

  // KDE brightness slider settings
  const kdeBrightnessSliderSettings = {
    //start: 0.0,
    min: 0.0,
    max: 15.0,
    step: 0.1,
    onChange: value => {
      setKdeBrightness(value);
    }
  };

  // KDE dropdown options
  const lookupKdeOptionsUrl = CONFIG.baseUrl + 'lookup/kde/options/'
  useEffect(() => {
    fetch(lookupKdeOptionsUrl)
      .then(res => res.json())
      .then(
        (result) => {
          setKdeOptions(result)
        },
        // Note: it's important to handle errors here
        // instead of a catch() block so that we don't swallow
        // exceptions from actual bugs in components.
        (error) => {
          console.log('Error loading KDE Options');
          console.log(error);
        }
      );
  }, [])


  function RenderHeatmapBinSize() {
    /* Bin Count slider
      NOTE: State handled by app.js and later passed to map URL. 
            Don't register with React Hook Form;
            keep separate as does not affect the
            filtering of data and would invalidate the cache
            if included in the form fields.
    */
    return (
      <Grid columns={2} divided>
        <Grid.Row>
          <Grid.Column>
            <Slider discrete
              value={numBins}
              color='blue'
              settings={numBinsSlider}
            />
          </Grid.Column>
          <Grid.Column>
            <Label circular>{numBins}</Label>
          </Grid.Column>
        </Grid.Row>
      </Grid>
    );
  }

  function RenderHeatmapBrightness() {
    /* Brightness slider 
           NOTE: State handled by app.js and later passed to map URL. 
                 Don't register with React Hook Form;
                 keep separate as does not affect the
                 filtering of data and would invalidate the cache
                 if included in the form fields.
       */
    return (
      <Grid columns={2} divided>
        <Grid.Row>
          <Grid.Column>
            <Slider discrete
              value={brightness}
              color='blue'
              settings={brightnessSlider}
            />
          </Grid.Column>
          <Grid.Column>
            <Label circular>{brightness}</Label>
          </Grid.Column>
        </Grid.Row>
      </Grid>
    );
  }

  function RenderRelativeMode() {

    // Loading...
    if (kdeOptions == null) {
      return (
        <p>Loading relative mode options...</p>
      );
    }

    // Check result has required property
    if (!(kdeOptions.hasOwnProperty('kde_relative_mode'))) {
      return (
        <p>Error: kde_relative_mode' not found in KDE options.</p>
      );
    }
    let relativeModeOptions = kdeOptions['kde_relative_mode'];

    // Render dropdown
    return (
      <Grid columns={2}>
        <Grid.Row>
          <Grid.Column>
            <Dropdown
              placeholder='Select relative mode...'
              fluid={false}
              selection
              options={relativeModeOptions}
              onChange={(ev, data) => setKdeRelativeMode(data.value.toString())}
              defaultValue={kdeRelativeMode}
            />
          </Grid.Column>
        </Grid.Row>
      </Grid>
    );
  }

  function RenderRelativeSelection() {

    // Loading...
    if (kdeOptions == null) {
      return (
        <p>Loading relative selection options...</p>
      );
    }

    // Check result has required property
    if (!(kdeOptions.hasOwnProperty('kde_relative_selection'))) {
      return (
        <p>Error: kde_relative_selection' not found in KDE options.</p>
      );
    }
    let relativeSelectionOptions = kdeOptions['kde_relative_selection'];

    // Render dropdown
    return (
      <Grid columns={2}>
        <Grid.Row>
          <Grid.Column>
            <Dropdown
              placeholder='Select relative selection...'
              fluid={false}
              selection
              options={relativeSelectionOptions}
              onChange={(ev, data) => setKdeRelativeSelection(data.value.toString())}
              defaultValue={kdeRelativeSelection}
            />
          </Grid.Column>
        </Grid.Row>
      </Grid>
    );
  }

  function RenderColormapSelection() {

    // Loading...
    if (kdeOptions == null) {
      return (
        <p>Loading colour map options...</p>
      );
    }

    // Check result has required property
    if (!(kdeOptions.hasOwnProperty('kde_colormap'))) {
      return (
        <p>Error: kde_colormap' not found in KDE options.</p>
      );
    }
    let colormapOptions = kdeOptions['kde_colormap'];

    // Render dropdown
    return (
      <Grid columns={2}>
        <Grid.Row>
          <Grid.Column>
            <Dropdown
              placeholder='Select colour map...'
              fluid={false}
              selection
              options={colormapOptions}
              onChange={(ev, data) => setKdeColormap(data.value.toString())}
              defaultValue={kdeColormap}
            />
          </Grid.Column>
          <Grid.Column>
            <Checkbox
              className='semantic-checkbox-adjust'
              toggle
              label='Invert'
              onChange={(ev, data) => setKdeColormapInvert(data.checked)}
              defaultChecked={kdeColormapInvert}
            />
          </Grid.Column>
        </Grid.Row>
      </Grid>
    );
  }

  function RenderRowName() {

    // Loading...
    if (facetRowValues == null || facetRowValues.length == 0 || facetRowValues[0] == '') {
      return (
        <p>No dataset loaded.</p>
      );
    }

    // Build options
    let facetRowOptions = [];
    facetRowOptions.push({ key: '', text: '', value: '' }); // blank option
    for (let i = 0; i < facetRowValues.length; i++) {
      const item = {
        key: facetRowValues[i],
        text: facetRowValues[i],
        value: facetRowValues[i],
      }
      facetRowOptions.push(item);
    }

    // Render dropdown
    return (
      <Grid columns={2}>
        <Grid.Row>
          <Grid.Column>
            <Dropdown
              placeholder='Select row value...'
              fluid={false}
              selection
              options={facetRowOptions}
              onChange={(ev, data) => setKdeRowName(data.value.toString())}
              defaultValue={kdeRowName}
            />
          </Grid.Column>
        </Grid.Row>
      </Grid>
    );
  }

  function RenderColumnName() {

    // Loading...
    if (facetColValues == null || facetColValues.length == 0 || facetColValues[0] == '') {
      return (
        <p>No dataset loaded.</p>
      );
    }

    // Build options
    let facetColOptions = [];
    facetColOptions.push({ key: '', text: '', value: '' }); // blank option
    for (let i = 0; i < facetColValues.length; i++) {
      const item = {
        key: facetColValues[i],
        text: facetColValues[i],
        value: facetColValues[i],
      }
      facetColOptions.push(item);
    }

    // Render dropdown
    return (
      <Grid columns={2}>
        <Grid.Row>
          <Grid.Column>
            <Dropdown
              placeholder='Select column value...'
              fluid={false}
              selection
              options={facetColOptions}
              onChange={(ev, data) => setKdeColumnName(data.value.toString())}
              defaultValue={kdeColumnName}
            />
          </Grid.Column>
        </Grid.Row>
      </Grid>
    );
  }

  function RenderKDESize() {
    /* KDE kernel bandwidth slider
           NOTE: State handled by app.js and later passed to map URL. 
                 Don't register with React Hook Form;
                 keep separate as does not affect the
                 filtering of data and would invalidate the cache
                 if included in the form fields.
     */
    return (
      <Grid columns={2} divided >
        <Grid.Row>
          <Grid.Column>
            <Slider
              value={kdeBandwidth}
              color='blue'
              settings={kdeBandwidthSliderSettings}
            />
          </Grid.Column>
          <Grid.Column>
            <Label circular>{kdeBandwidth.toFixed(1)}</Label>
          </Grid.Column>
        </Grid.Row>
      </Grid >
    );
  }

  function RenderKDEBrightness() {
    /* KDE kernel brightness slider
       NOTE: State handled by app.js and later passed to map URL. 
             Don't register with React Hook Form;
             keep separate as does not affect the
             filtering of data and would invalidate the cache
             if included in the form fields.
   */
    return (
      <Grid columns={2} divided>
        <Grid.Row>
          <Grid.Column>
            <Slider
              value={kdeBrightness}
              color={'blue'}
              settings={kdeBrightnessSliderSettings}
            />
          </Grid.Column>
          <Grid.Column>
            <Label circular>{kdeBrightness.toFixed(1)}</Label>
          </Grid.Column>
        </Grid.Row>
      </Grid>
    );
  }

  function renderLayout() {
    return (
      <div>
        <Form>
          <Divider horizontal><Header as='h4'><Icon className={'bi bi-grid-fill'} />Heatmap</Header></Divider>
          {/* Need to call as function, otherwise slider doesn't slide..? */}
          <Form.Field>
            <label>Brightness</label>
            {RenderHeatmapBrightness()}
          </Form.Field>
          <Form.Field>
            <label>Bin size</label>
            {RenderHeatmapBinSize()}
          </Form.Field>
          <p><br /></p>
          <Divider horizontal><Header as='h4'><Icon className={'bi bi-record-circle-fill'} />KDE</Header></Divider>
          {/* Need to call as function, otherwise slider doesn't slide..? */}
          <Form.Field>
            <label>Brightness</label>
            {RenderKDEBrightness()}
          </Form.Field>
          <Form.Field>
            <label>Bandwidth</label>
            {RenderKDESize()}
          </Form.Field>
          <Form.Field>
            <label>Colour map</label>
            <RenderColormapSelection />
          </Form.Field>
          <Form.Field>
            <label>Relative mode</label>
            <RenderRelativeMode />
          </Form.Field>
          <Form.Field>
            <label>Relative selection</label>
            <RenderRelativeSelection />
          </Form.Field>
          <Form.Field>
            <label>Row value</label>
            <RenderRowName />
          </Form.Field>
          <Form.Field>
            <label>Column value</label>
            <RenderColumnName />
          </Form.Field>
        </Form>
      </div>
    );
  }

  // Final render
  return (
    renderLayout()
  );
}

export default React.memo(KDESettings)