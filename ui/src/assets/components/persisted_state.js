// React component to save and retrieve state to local storage
// Use instead of useState()

// Adapted from https://dev.to/selbekk/persisting-your-react-state-in-9-lines-of-code-9go
// 23/04/2021 

// Will not work in Private mode on Safari, remember to catch exceptions from setItem():
// https://developer.mozilla.org/en-US/docs/Web/API/Storage/setItem
import React, { useEffect } from 'react'

export default function usePersistedState(key, defaultValue) {
  const [state, setState] = React.useState(
    () => JSON.parse(localStorage.getItem(key)) || defaultValue
  );
  useEffect(() => {
    localStorage.setItem(key, JSON.stringify(state));
  }, [key, state]);
  return [state, setState];
}