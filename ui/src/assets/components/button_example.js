/* Simple React Button (for testing) */

import React from "react"
import { Button } from 'semantic-ui-react'

const ButtonExample = React.memo(React.forwardRef((props, ref) =>
  <MyButtonContainer ref={ref} content='Click Here'>
  </MyButtonContainer>
), true);

const MyButtonContainer = React.memo(React.forwardRef((props, ref) => {
  const someConstant = 1;
  return (
    <Button ref={ref} content={props.content}>
    </Button>
  );
}), true);

export default ButtonExample;