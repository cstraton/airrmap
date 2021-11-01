// ROI Report: VDJ distribution plot
// Processes the data returned from the application server and
// renders the plots.
// Facets will be rendered same order as rendered tiles.

import React from "react"
import Plot from 'react-plotly.js';

export default function ROIVDJ({ report, section_name, facetRowValues, facetColValues, setAppStatus }) {

  // ** Validation **

  // Check report is available
  if (report == null ||
    report == undefined ||
    (!report.hasOwnProperty(section_name))) {
    return null;
  }

  // Check facetRowValues and facetColValues are available
  if (facetRowValues == undefined || facetColValues == undefined) {
    return null;
  }

  // ** Init **

  // Vars
  const keyDelim = '|'

  // Build facet properties
  // Key = 'rowValue|colValue'
  let facetItems = {}
  let facetRowIndex = 1;
  let subplotIndex = 0;
  let subplots = [];
  for (const rowValue of facetRowValues) {

    // Store subplots for the row
    let facetColIndex = 1;
    let rowSubplots = []

    // Loop columns
    for (const colValue of facetColValues) {

      // Create item
      subplotIndex++;
      const item = {
        key: rowValue.toString() + keyDelim + colValue.toString(),
        name: rowValue.toString() + '<br />' + colValue.toString(),
        subplotIndex: subplotIndex,
        facetRowIndex: facetRowIndex,
        facetColIndex: facetColIndex,
        xaxis: 'x' + (facetColIndex == 1 ? '' : facetColIndex.toString()), // share x axis for each column
        yaxis: 'y' + (facetRowIndex == 1 ? '' : facetRowIndex.toString()), // share y axis for each row
        xref: 'x' + (facetColIndex == 1 ? '' : facetColIndex.toString()) + ' domain', // annotation / title
        yref: 'y' + (facetRowIndex == 1 ? '' : facetRowIndex.toString()) + ' domain', // annotation / title
        traceItem: null, // placeholder
        annotationItem: null // placeholder
      }

      // Store
      // Subplots format: https://plotly.com/javascript/subplots/
      rowSubplots.push(item['xaxis'] + item['yaxis'])
      //subplots.push(item['xaxis'] + item['yaxis'])
      facetItems[item['key']] = item;
      facetColIndex++;
    }

    // Add row of subplots, and increment
    subplots.push(rowSubplots);
    facetRowIndex++;

  }

  // Get the report and properties
  const reportSub = report[section_name];
  const reportName = reportSub['name'];
  const reportTitle = reportSub['title'];
  const reportType = reportSub['report_type'];
  const reportXLabel = reportSub['x_label'];
  const reportYLabel = reportSub['y_label'];
  const reportFacetData = reportSub['data']; // list, one item per facet

  // Build up each facet plot from the data
  for (const facetData of reportFacetData) {

    // Get the data values
    const facetRowValue = facetData['facet_row_value'].toString(); // e.g. 'Fv_volunteer'
    const facetColValue = facetData['facet_col_value'].toString(); // e.g. 'after-Day-1'
    const groupValues = facetData['group_values'];        // e.g. [2, 3, 4]
    const measureValues = facetData['measure_values'];         // e.g. [10, 20, 30]
    const measurePcts = facetData['measure_pcts'];   // e.g. [20.0, 40.0, 40.0]

    // Check we have everything
    if (!(facetRowValue && facetColValue && groupValues && measureValues && measurePcts)) {
      setAppStatus(
        'Not all expected values received in report.',
        'error',
        false
      );
      return null;
    }

    // Properties
    const facetKey = facetRowValue + keyDelim + facetColValue;
    const facetItem = facetItems[facetKey];
    const facetIndex = facetItem['subplotIndex'].toString();

    // Define the subplot
    const trace = {
      x: groupValues,
      y: measurePcts,
      xaxis: facetItem['xaxis'],
      yaxis: facetItem['yaxis'],
      type: 'bar',
      name: facetItem['name']
    }

    // Define the subplot title (annotation workaround)  
    // Plotly.js doesn't currently support 
    // subplot titles https://github.com/plotly/plotly.js/issues/2746
    // Adapted from codepen: https://codepen.io/nicolaskruchten/pen/ExZNPbz
    const subplotTitle = {
      text: facetItem['name'],
      showarrow: false,
      x: 0.5,
      xref: facetItem['xref'],
      yref: facetItem['yref'],
      y: 1.2,
    }

    // Store
    facetItem['traceItem'] = trace;
    facetItem['annotationItem'] = subplotTitle;

  }

  // Build 1D array of trace items and annotations for plotly
  let traceItems = []
  let annotationItems = []
  for (const facetRowValue of facetRowValues) {
    for (const facetColValue of facetColValues) {
      const facetKey = facetRowValue.toString() + keyDelim + facetColValue.toString();
      const facetItem = facetItems[facetKey];
      traceItems.push(facetItem['traceItem']);
      annotationItems.push(facetItem['annotationItem']);
    }
  }

  // Define the grid
  let layout = {
    //autosize: true,
    //width: 1500,
    //height: 800,
    //title: reportTitle,
    xaxis: {
      type: 'category',
      title: ''
    },
    grid: {
      rows: facetRowValues.length,
      columns: facetColValues.length,
      subplots: subplots,
      roworder: 'top to bottom',
      pattern: 'independent' // coupled doesn't appear to work?
    },
    annotations: annotationItems,
    showlegend: false // hide right interactive legend
  };

  // Config
  let plotConfig = {
    displayModeBar: false,
    responsive: true
  }

  // Render the plot
  return (
    <Plot
      config={plotConfig}
      data={traceItems}
      layout={layout}
      useResizeHandler={true}
      // style: see also roi_report.css
      style={{ width: "100%", height: "100%" }}
    />
  )
}