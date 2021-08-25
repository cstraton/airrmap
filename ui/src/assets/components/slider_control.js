// Wrap react-semantic-ui-range for <Controller /> use
// with React Hook Form.
// react-semantic-ui-range: 
// https://github.com/iozbeyli/react-semantic-ui-range#readme

import React from 'react'
import {useController, useForm} from 'react-hook-form'
import { Slider } from 'react-semantic-ui-range'
//import 'semantic-ui-css/semantic.min.css' // Already loads in app.js

export default function SliderControl({ control, name, defaultValue, setValueFn }) {

  // React Hook Form <Controller /> integration
  // https://react-hook-form.com/api/usecontroller
  const {
    field: {ref, ...inputProps },
    fieldState: { invalid, isTouched, isDirty },
    formState: { touchedFields, dirtyFields }
  } = useController({
    name,
    control,
    rules: { required: true },
    defaultValue: defaultValue
  });

  // Settings
  const settings = {
    start: 0,
    min: 0,
    max: 4,
    step: 1,
    onChange: value => {
      setValueFn(name, value);
    }
  }

  return (
    <Slider
          {...inputProps}
          inputRef={ref}
          discrete
          settings={settings}
        />
    );
}