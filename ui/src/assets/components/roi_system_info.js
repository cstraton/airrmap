// ROI Report: System Information
// Processes the data returned from the application server 

import React, { useState } from "react"
import { Container, Header } from 'semantic-ui-react'

export default function ROISystemInformation({ report, section_name, setAppStatus }) {

  // ** Validation **

  // Check report is available
  if (report == null ||
    report == undefined ||
    (!report.hasOwnProperty(section_name))) {
    return null;
  }

  // Get the report and properties
  const reportSection = report[section_name];
  const reportDebugFormatted = JSON.stringify(reportSection, null, 4);

  // Render the report
  return (
    <Container fluid>
      {/*<Header as='h2'>System Information</Header>*/}
      <p>
        <pre>
        {reportDebugFormatted}
        </pre>
      </p>
    </Container>
  );
}

