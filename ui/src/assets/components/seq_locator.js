// Sequence locator interface

import React, { useEffect, useState, useRef } from "react"
import { Button, Container, Form, TextArea, Popup } from 'semantic-ui-react'
import usePersistedState from "./persisted_state";
import CONFIG from '../../../config.json';
//import './css/roi_report.css'

function SequenceLocator({ env_name, setAppStatus, submitCallback }) {

  // State and vars
  const [sequences, setSequences] = usePersistedState('store-seq-locator-sequences', '');
  const seqsCurrentValue = useRef(sequences); // store value when changing, but don't trigger redraw

  // Render layout
  function RenderLayout() {

    const seqsRef = useRef()

    // Submit handler
    const handleSubmit = (() => {
      console.log(seqsRef.current)
      let seq_list = seqsCurrentValue.current;
      setSequences(seq_list); // Triggers render  
      submitCallback(env_name, seq_list); // Pass the sequences back
    });

    // Render
    return (
      <Container fluid>
        <p>Environment: {env_name}</p>
        <Form>
          <Form.Field>
            <TextArea
              ref={seqsRef}
              onChange={(ev, data) => { seqsCurrentValue.current = data['value'] }}
              //onInput={(ev, {data}) => setSequences(data['value'])}
              placeholder='Enter sequences, one per row. Case sensitive. Should be the same format as the anchor sequences in the environment.'
              defaultValue={sequences}
              style={{ minHeight: 300 }}
            />
          </Form.Field>

          <Popup
            content={CONFIG.tooltips.sidebar.search.submit}
            mouseEnterDelay={CONFIG.tooltips.mouseEnterDelay}
            mouseLeaveDelay={CONFIG.tooltips.mouseLeaveDelay}
            trigger={
              <Button onClick={() => handleSubmit()}>Submit</Button>
            }
          />
        </Form>
      </Container>
    );
  }
  return (
    <RenderLayout />
  );
}

export default React.memo(SequenceLocator)