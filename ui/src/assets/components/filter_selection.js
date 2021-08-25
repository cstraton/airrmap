// Filter selection pane

// API dropdown should return array of dictionaries with key, text, value.
//
// Example:
// [
//  {"key": "key1", "text": "Text 1", "value": "Value 1"},
//  {"key": "key2", "text": "Text 2", "value": "Value 2"},
// ]

// Includes code adapted from:
// https://codesandbox.io/s/6j1760jkjk
// https://reactjs.org/docs/faq-ajax.html
// https://react.semantic-ui.com/modules/dropdown/#types-search-selection-two
// https://daveceddia.com/usereducer-hook-examples/


import React, { useState } from 'react'
import { useForm } from 'react-hook-form'
import { Button, Form, Header, Icon } from 'semantic-ui-react'
import LookupList from './lookup_list'
import SliderControl from './slider_control'
import { Slider } from 'react-semantic-ui-range'
import 'bootstrap-icons/font/bootstrap-icons.css';
import './css/filter_selection.css';
import CONFIG from '../../../config.json';


function FilterSelection({ submitHandler,
  brightness, setBrightness, numBins, setNumBins }) {

  // React Hook Form
  const { register, unregister, handleSubmit, control, watch, reset, setValue, getValues } = useForm(); //{ mode: "onBlur" }

  // Submit handler
  const onSubmit = data => {
    submitHandler(data)
  }

  // Filters selection - Add / Remove
  // Adapted from: https://codesandbox.io/s/6j1760jkjk
  const [indexes, setIndexes] = useState([]);
  const [counter, setCounter] = useState(0);

  // Add a single filter
  function filterAdd(e) {
    setIndexes(prevIndexes => [...prevIndexes, counter]);
    setCounter(prevCounter => prevCounter + 1);
    e.preventDefault(); // stop form from submitting
  }

  // Remove a single filter
  function filterRemove(e, index) {

    // Set indexes to all previous indexes except the removed one
    setIndexes(prevIndexes => [...prevIndexes.filter(item => item !== index)]);
    setCounter(prevCounter => prevCounter - 1);

    // Unregister fields from react-hook-form
    // so values don't continue being submitted
    unregister([
      `filters[${index}].filter_name`,
      `filters[${index}].filter_value`
    ],
      {
        keepValue: false,
        keepDefaultValue: false
      }
    );

    // Stop form from submitting
    e.preventDefault();
  }

  // Clear all filters
  function filterRemoveAll(e) {

    // Unregister from react-hook-form
    for (let i = 0; i < indexes.length; i++) {
      unregister([
        `filters[${i}].filter_name`,
        `filters[${i}].filter_value`
      ],
        {
          keepValue: false,
          keepDefaultValue: false
        }
      );
    }
    setIndexes([]);
    setCounter(0);

    // Stop form from submitting
    e.preventDefault();
  }

  // Num Bins Slider
  const numBinsSlider = {
    start: 0,
    min: 0,
    max: 4,
    step: 1,
    onChange: value => {
      setNumBins(value);
    }
  }

  // Brightness slider settings
  const brightnessSlider = {
    start: 1.0,
    min: 1.0,
    max: 200,
    step: 10.0,
    onChange: value => {
      setBrightness(value);
    }
  };

  // Define default values
  // TODO: Change
  const defaultValues = {
    env_name: '',
    facet_row: 'file.Author',
    facet_col: 'file.Author'
  }

  // Need to register the default values with React Hook Forms
  // after React renders, otherwise they will be null.
  // https://github.com/react-hook-form/react-hook-form/issues/1150
  //useEffect(() => {
  //  reset(defaultValues);
  //}, []); // default [], only call once.

  // Watch first filtername
  //console.log(watch("filters[0].filterValue"))

  // Selected filters
  /*
  const [selectedItems, dispatch] = useReducer((state, action) => {
    switch (action.type) {

      case 'add':
        return [
          ...state,
          {
            filterId: state.length,
            filterName: action.name,
            filterValue: action.value
          }
        ];

      case 'remove':
        // keep item
        return state.filter((_, index) => index != action.index);


      default:
        return state;
    }
  }, [{ filterId: 1, filterName: '', filterValue: '' }]); // default value

*/

  function setStatus(status) {
    console.log(status);
  }

  function RenderLookupEnv() {
    return (
      <LookupList
        listID='env-name1-list'
        baseUrl={CONFIG.baseUrl + 'lookup/env/'}
        fnTransform={null}
        fnStatus={setStatus}
      />
    );
  }

  const RenderLookupFilterField = LookupList => props => (
    <LookupList
      listID={'field-list1'}
      baseUrl={CONFIG.baseUrl + 'lookup/env/abc/fields/'}
      fnTransform={null}
      fnStatus={setStatus}
    />
  );

  function renderEnvironment() {
    return (
      <React.Fragment>
        {/* Environment Name */}
        <Header as='h5'><Icon className={'bi bi-filter'} />Environment</Header>
        <Form.Group name='env-name' widths='equal'>
          <Form.Field>
            {/*<label htmlFor='env_name'><Icon className='bi bi-filter' /> Environment Name</label>*/}
            <input
              type={'text'}
              name={'env_name'}
              list={'env-name1-list'}
              defaultValue={defaultValues['env_name']}
              placeholder={'Environment name...'}
              {...register('env_name')}
            />
          </Form.Field>
          <Button type='submit'>Submit</Button>
        </Form.Group>
      </React.Fragment>
    );
  }

  function renderFacets() {
    return (
      <React.Fragment>
        {/* Facets */}

        {/* Enabled */}
        {/*
      <Form.Checkbox toggle 
        name='facet_enabled'
        {...register('facet_enabled')}
      />
      */}

        <Header as='h5'><Icon className={'bi bi-layout-split'} />Facets | row, column</Header>
        {/*<label htmlFor='facets'><Icon className='bi bi-layout-split' /> Facets | row, column</label>*/}
        <Form.Group name='facets' widths='equal'>
          <Form.Field>
            <input
              type={'text'}
              name={'facet_row'}
              list={'field-list1'}
              defaultValue={defaultValues['facet_row']}
              placeholder={'Row field...'}
              {...register('facet_row')}
            />
          </Form.Field>
          <Form.Field>
            <input
              type={'text'}
              name={'facet_col'}
              list={'field-list1'}
              defaultValue={defaultValues['facet_col']}
              placeholder={'Column field...'}
              {...register('facet_col')}
            />
          </Form.Field>
        </Form.Group>
      </React.Fragment>
    );
  }

  function renderBinSize() {
    return (
      <React.Fragment>
        {/* Bin Count slider
            NOTE: State handled by app.js and later passed to map URL. 
                  Don't register with React Hook Form;
                  keep separate as does not affect the
                  filtering of data and would invalidate the cache
                  if included in the form fields.
        */}
        <Header as='h5'><Icon className='bi bi-grip-horizontal' />Bin size</Header>
        <Slider discrete
          value={numBins}
          color='blue'
          settings={numBinsSlider}
        />
      </React.Fragment>
    );
  }

  function renderBrightness() {
    return (
      <React.Fragment>
        {/* Brightness slider 
            NOTE: State handled by app.js and later passed to map URL. 
                  Don't register with React Hook Form;
                  keep separate as does not affect the
                  filtering of data and would invalidate the cache
                  if included in the form fields.
        */}
        <Header as='h5'><Icon className={brightness == 1.0 ? 'bi bi-lightbulb-off' : 'bi bi-lightbulb'} />Brightness</Header>
        <Slider discrete
          value={brightness}
          color='blue'
          settings={brightnessSlider}
        />
      </React.Fragment>
    );
  }

  function renderFilters() {
    return (
      <React.Fragment>
        <Header as='h5'><Icon className='bi bi-funnel' />Filter</Header>

        {indexes.map(index => {
          let fieldName = `filters[${index}]`; // If updating, update removeFilter(). Use let not const.
          return (
            <Form.Group name={fieldName} key={fieldName}>
              {/*
              <Form.Checkbox
                name ={`${fieldName}.filter_enabled`}
                {...register(`${fieldName}.filter_enabled`)}
              />
              */}

              <Form.Input
                key={'filterName'}
                name={`${fieldName}.filter_name`}
                list='field-list1'
                //ref={register}
                placeholder='Field...'
                {...register(`${fieldName}.filter_name`)}
              />

              <Form.Input
                key={'filterValue'}
                name={`${fieldName}.filter_value`}
                //ref={register}
                placeholder='Criteria...'
                {...register(`${fieldName}.filter_value`)}
              />

              {/* Remove single filter button */}
              <Button icon
                key={'filterRemove'}
                onClick={(e) => filterRemove(e, index)}>
                <Icon name='close' />
              </Button>

            </Form.Group>
          );
        })}

        <Button icon onClick={filterAdd}><Icon name='add' /></Button>
        <Button icon onClick={filterRemoveAll}><Icon name='close' /></Button>

      </React.Fragment>
    );
  }


  function renderLayout() {
    //console.log('Rendering again...');
    return (
      <div>
        <Form size={'small'} onSubmit={handleSubmit(onSubmit)}>
          {/*<RenderLookupEnv /> */}
          <LookupList
            listID='env-name1-list'
            baseUrl={CONFIG.baseUrl + 'lookup/env/'}
            fnTransform={null}
            fnStatus={setStatus}
          />
          <LookupList
            listID={'field-list1'}
            baseUrl={CONFIG.baseUrl + 'lookup/env/abc/fields/'}
            fnTransform={null}
            fnStatus={setStatus}
          />
          {renderEnvironment()}
          {renderFacets()}
          {renderBinSize()}
          {renderBrightness()}
          {renderFilters()}
        </Form>
      </div>
    );
  }

  // Final render
  return (
    renderLayout()
  );

}

export default React.memo(FilterSelection)

