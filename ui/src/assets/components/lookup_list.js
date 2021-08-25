// Lookup list for use with Input controls
// Returns a <datalist /> component that other controls can reference.
// Calls API

import React, { useState, useEffect } from 'react'
import { v4 as uuidv4 } from 'uuid'

//const LookupList = React.memo(({ listID, baseUrl, fnTransform, fnStatus }) => {
function LookupList ({ listID, baseUrl, fnTransform, fnStatus })  {

  // Vars
  const [error, setError] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const [items, setItems] = useState([]);

  // API call
  useEffect(() => {
    let mounted = true;
    fetch(baseUrl)
      .then(res => res.json())
      .then(
        (result) => {
          // Transform if provided
          if (fnTransform !== undefined && fnTransform !== null) {
            result = fnTransform(result);
          }
          setItems(result);
          setIsLoaded(true);
        },
        // Note: it's important to handle errors here
        // instead of a catch() block so that we don't swallow
        // exceptions from actual bugs in components.
        (error) => {
          setError(error);
          setIsLoaded(true);
        }
      );
    return () => mounted = false; // Stop continuous calls
  }, []); // default [], only call once.

  // Error
  if (error) {
    if (fnStatus !== undefined && fnStatus !== null) {
      fnStatus('Error: ' + error.message);
    }
  }

  // Loaded
  if (!{ isLoaded }) {
    if (fnStatus !== undefined && fnStatus !== null) {
      fnStatus('Loading...')
    }
    return null;
  }

  // Render
  return (
    <datalist key={listID} id={listID}>
      {items.map(item => {
        return (
          <option key={uuidv4()} value={item}></option>
        );
      })}
    </datalist>
  );

}

export default LookupList