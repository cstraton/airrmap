// Semantic UI grid of leaflet maps

import React  from 'react'
import { Table } from 'semantic-ui-react'
import LeafletItem from './leaflet_item.js'
import './css/leaflet_grid.css';

// LeafletGrid component
function LeafletGrid ( {
    mapController,
    facetRowValues,
    facetColValues,
    queryReport,
    mapStatsEnabled
    }) {

  // If facet row/column was specified, the server will have provided a list
  // of values in the previous query result. A separate map will be created for
  // each row/col value, and the map uri will fetch the related subset of data.
  // We need a minimum of one row and one col element, set to '' if not supplied.
  facetRowValues = facetRowValues || [];
  const rowValues = facetRowValues.length > 0 ? facetRowValues : [''];
  facetColValues = facetColValues || [];
  const colValues = facetColValues.length > 0 ? facetColValues : [''];

  // Build config for each map (row, col)
  // Url format:
  // e.g. http://localhost:5000/tiles/imagery/<zoom>/<x>/<y>?a=1&q=<query_filters> ...
  //      ... &fr=record.Subject&fc=record.Longitudinal}> ...
  //      ...&v1min=1.0&v1max=2880
  // Array format:
  //      [[1, 2, 3], 
  //       [4, 5, 6], 
  //       [7, 8, 9]]
  // Map item: 
  // {'mapKey': 'leafletItem-r1-c2', 'mapUrl': 'http://...'}
  let mapItemArray = []
  for (let iRow = 0; iRow < rowValues.length; iRow++) {
    let colMapItems = [];
    for (let iCol = 0; iCol < colValues.length; iCol++) {
      const mapKey = 'leafletItem-r' + iRow + '-c' + iCol;
      const mapRowValue = rowValues[iRow];
      const mapColValue = colValues[iCol];
      
      // Get statistics from the query report (min, max etc.)
      let mapStats = null;
      if (queryReport.hasOwnProperty('value1_facet_stats')) {
        const reportFacetKeyDelim = queryReport['facet_key_delim'];
        const reportFacetKey = mapRowValue.toString() + reportFacetKeyDelim + mapColValue.toString();
        mapStats = queryReport['value1_facet_stats'][reportFacetKey];
      }
      const mapItem = { 'mapKey': mapKey, 'r': rowValues[iRow], 'c': colValues[iCol], 'stats': mapStats };
      colMapItems.push(mapItem);
    }
    mapItemArray.push(colMapItems);
  }

  // Render
  // Uses indexes for keys as layout should be fixed until next query result.
  console.log('Render being called...')
  return (
    <Table celled inverted className='leaflet-grid-table'>
      <Table.Body>
        {mapItemArray.map((mapRowItems, iRow) => (
          <Table.Row key={iRow} color='black'>
            {mapRowItems.map((mapItem, iCol) => (
              <Table.Cell key={mapItem['c']} className='leaflet-grid-cell'>
                <LeafletItem
                  key={iCol}
                  id={mapItem['mapKey']} // need it as a prop, 'key' will be undefined if trying to read.
                  mapController={mapController}
                  mapLabel={mapItem['r'] + ', ' + mapItem['c']}
                  iRow={iRow}
                  iCol={iCol}
                  facetRowValue={mapItem['r']}
                  facetColValue={mapItem['c']}
                  mapStatsEnabled={mapStatsEnabled}
                  stats={mapItem['stats']}
                >
                </LeafletItem>
              </Table.Cell>
            ))}
          </Table.Row>
        ))}
      </Table.Body>
    </Table>
  );
}

export default React.memo(LeafletGrid)