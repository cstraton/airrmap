// Environment selection and filters panel

import React, { Component } from 'react'
import {
  Button,
  Dropdown,
  Form,
  Header,
  Icon,
  Input
} from 'semantic-ui-react'
import CONFIG from '../../../config.json';

class FilterSelection2 extends Component {

  // Init state
  //state = {
  //  envSelected: null,
  //  _envList: null,
  //  _envFieldList: null,
  //  filters: []
  // }

  // Constructor
  // Load state from local storage or set defaults
  constructor() {
    super();

    // Set up state
    this.state = {
      envSelected: null,
      _envList: null,
      _envFieldList: null,
      filters: []
    }

    // Load state if exists
    const storedStateJson = window.localStorage.getItem('filter-state')
    console.log(storedStateJson)
    if (storedStateJson) {
      const storedState = JSON.parse(storedStateJson)
      for (let x in storedState) {
        this.state[x] = storedState[x]
      }
    }
  }

  // Override setState
  // Write to local storage
  // State arg will only be the property
  // that was updated.
  setState(state) {

    // Get existing state + incoming new state
    // ('...' spread op to convert objects to args)
    const currentState = { ...this.state, ...state }

    // Make a copy without private props (_)
    // (e.g. leave dropdown list values)
    let localStorageState = {}
    for (let propName in currentState) {
      if (propName[0] !== '_') {
        localStorageState[propName] = currentState[propName]
      }
    }

    // Store all properties
    window.localStorage.setItem('filter-state', JSON.stringify(localStorageState));

    // Pass state up
    super.setState(state)
  }

  // Handle real-time changes to form fields. name: value
  handleChange = (e, { name, value }) => this.setState({ [name]: value })

  // Submit handler
  handleSubmit = (e) => {

    // Build list of filters
    let submitFilters = []
    const filters = this.state.filters
    for (var i = 0; i < filters.length; i++) {
      const filterItem = filters[i]
      const filterItemSubmit = {}
      filterItemSubmit['filter_name'] = filterItem.filterName
      filterItemSubmit['filter_value'] = filterItem.filterValue
      submitFilters.push(filterItemSubmit)
    }

    // Build submit data
    let submitData = {}
    submitData['env_name'] = this.state.envSelected
    submitData['facet_row'] = this.state.facetRow
    submitData['facet_col'] = this.state.facetCol
    submitData['filters'] = submitFilters
    console.log(submitData)

    // Submit
    this.props.submitHandler(submitData);

  }

  // Add filter
  filterAdd = (e) => {
    e.preventDefault()
    this.setState({
      filters: this.state.filters.concat([{
        filterName: '',
        filterValue: ''
      }])
    })
  }

  // Change filter (don't mutate array, create copy of item)
  filterChange = (e, index, isFilterName) => {
    e.preventDefault()
    console.log(e)
    this.setState({
      filters: this.state.filters.map((filterItem, _index) => {
        if (_index !== index) return filterItem;
        // creates a new object, with fields from filterItem
        // and then set relevant property.
        if (isFilterName) {
          // e will be dropdown div, use .innerText
          return { ...filterItem, filterName: e.target.innerText }
        } else {
          return { ...filterItem, filterValue: e.target.value }
        }
      })
    });
  }

  // Remove filter
  filterRemove = (e, index) => {
    // Set state
    e.preventDefault()
    this.setState({
      filters: this.state.filters.filter((x, _index) => _index !== index)
    })
  }

  // Remove all filters
  filterRemoveAll = (e) => {
    e.preventDefault();
    this.setState({
      filters: []
    })
  }

  // Load environment list dropdown options
  loadEnvList = () => {
    var lookupEnvListUrl = CONFIG.baseUrl + 'lookup/env/'
    fetch(lookupEnvListUrl)
      .then(res => res.json())
      .then(
        (result) => {
          this.setState({ _envList: result })
        },
        // Note: it's important to handle errors here
        // instead of a catch() block so that we don't swallow
        // exceptions from actual bugs in components.
        (error) => {
          console.log('Error loading environment list.');
          console.log(error);
        }
      );
  }

  // Load environment fields
  loadEnvFields = (envSelected) => {

    if (!(envSelected)) {
      this.setState({ _envFieldList: null })
      return;
    }

    var lookupEnvFieldsUrl = CONFIG.baseUrl + 'lookup/env/' + envSelected + '/fields/'
    fetch(lookupEnvFieldsUrl)
      .then(res => res.json())
      .then(
        (result) => {
          this.setState({ _envFieldList: result })
        },
        (error) => {
          console.log('Error loading environment fields.');
          console.log(error);
          this.setState({ _envFieldList: null })
        }
      );
  }


  // Functions to run after render
  componentDidMount() {
    this.loadEnvList()
    this.loadEnvFields(this.state.envSelected)
  }

  // Functions to call when state changes
  // Must always use condition, otherwise infinite loop
  componentDidUpdate(prevProps, prevState, snapshot) {
    if (prevState.envSelected !== this.state.envSelected) {
      this.loadEnvFields(this.state.envSelected)
    }
  }


  // Render
  render() {
    const { envSelected, facetRow, facetCol, filters } = this.state
    return (
      <Form onSubmit={this.handleSubmit} size={'small'}>
        {/* Environment and submit */}
        <Header as='h5'><Icon className={'bi bi-filter'} />Environment</Header>
        <Form.Group>
          <Form.Field
            name='envSelected' // name important - must match state?
            value={envSelected}
            control={Dropdown}
            options={this.state._envList}
            placeholder='Select environment...'
            fluid={true}
            search
            selection
            onChange={this.handleChange}
          />
          <Form.Button content='Submit' loading={this.props.appStatusLoading}/>
        </Form.Group>

        {/* Facet row/column */}
        <Header as='h5'><Icon className={'bi bi-layout-split'} />Facets | row, column</Header>
        <Form.Group name='facets' widths='equal'>
          <Form.Field
            name='facetRow'
            value={facetRow}
            control={Dropdown}
            options={this.state._envFieldList}
            placeholder='Row field...'
            search
            selection
            onChange={this.handleChange}
          />
          <Form.Field
            name='facetCol'
            value={facetCol}
            control={Dropdown}
            options={this.state._envFieldList}
            placeholder='Column field...'
            search
            selection
            onChange={this.handleChange}
          />
        </Form.Group>

        {/* Filters */}
        <Header as='h5'><Icon className='bi bi-funnel' />Filter</Header>
        {this.state.filters.map((filterItem, index) => (
          <Form.Group widths='equal' key={index}>
            <Form.Field
              key='filter_name'
              control={Dropdown}
              options={this.state._envFieldList}
              value={filterItem.filterName}
              search
              selection
              placeholder='Field...'
              onChange={(e) => this.filterChange(e, index, true)}
            />
            <Form.Field
              key='filter_value'
              control={Input}
              value={this.state.filters[index].filterValue}
              placeholder='Criteria...'
              onChange={(e) => this.filterChange(e, index, false)}
            />
            {/* Remove single filter button */}
            <Button icon
              key={'filterRemove'}
              onClick={(e) => this.filterRemove(e, index)}>
              <Icon name='close' />
            </Button>

          </Form.Group>
        ))}
        <Button icon onClick={this.filterAdd}><Icon name='add' /></Button>
        <Button icon onClick={this.filterRemoveAll}><Icon name='close' /></Button>
      </Form>
    )
  }
}

export default FilterSelection2