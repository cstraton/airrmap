// Region of Interest (ROI) report

import React, { useState } from "react"
import { Container, Image, Tab, Grid, Statistic, Header, List } from 'semantic-ui-react'
import ROICDR3Length from './roi_cdr3length.js'
import ROIVDJ from './roi_vdj.js'
import ROIReportInformation from './roi_report_info.js'
import ROISystemInformation from './roi_system_info.js'
import './css/roi_report.css'

function ROIReport({ report, facetRowValues, facetColValues, setAppStatus }) {

  // Vars
  const EmptyComponent = (props) => { return null };
  let LogoRender = EmptyComponent;
  let SeqsRender = EmptyComponent;
  let StatusMessage = EmptyComponent;

  // If nothing
  if (report == null) {
    StatusMessage = (props) => {
      return (<p>Make a selection</p>);
    }
  }

  // Check for logo
  //if (report !== null && report.hasOwnProperty('logo')) {
  //  LogoRender = (props) => { return <Image src={report['logo']} /> };
  //}

  // Check for logos
  if (report !== null && report.hasOwnProperty('logo')) {
    let logoList = report['logo']

    LogoRender = (props) => {
      return (
        <Container>
        <Grid divided='vertically'>
          {logoList.map((item, index) => {
            const logoImage = item['logo_img'];
            const regionName = item['region_name'];
            const seqsTop = item['seqs_top'];
            const seqsBottom = item['seqs_bottom'];
            const seqs_unique_count = item['seqs_unique_count'];
            return (
              <Grid.Row>
                <Grid.Column width={8} centered>
                  <React.Fragment>
                    <Header textAlign='center' size='small'>{regionName} ({seqs_unique_count.toLocaleString()} unique)</Header>
                    <Image key='logo' src={logoImage} />
                  </React.Fragment>
                </Grid.Column>
                <Grid.Column width={5}>
                  <Container textAlign='center'>
                    <Header size='tiny' textAlign='center'>Top {seqsTop.length}</Header>
                    <List divided verticalAlign='middle'> 
                    {seqsTop.map((seqItem, seqIndex) => {
                      const seqString = seqItem[0];
                      const seqCount = seqItem[1];
                      return (
                        <List.Item key={seqIndex} className='roi-select-seq-strings-ol'>{seqString} ({seqCount.toLocaleString()})</List.Item>
                      );
                    })}
                    </List>
                  </Container>
                </Grid.Column>
                {/*
                <Grid.Column width={4}>
                  <Header size='tiny'>Bottom</Header>
                  {seqsBottom.map((seqItem, seqIndex) => {
                    const seqString = seqItem[0];
                    const seqCount = seqItem[1];
                    return (
                      <ol key={seqIndex} class='roi-select-seq-strings-ol'>{seqString} ({seqCount})</ol>
                    );
                  })}
                </Grid.Column>
                */}

              </Grid.Row>
            )
          })}
        </ Grid>
        </Container>
      )
    }
  }

  // Check for sequences
  if (report !== null && report.hasOwnProperty('seqs')) {
    let seq_text = []
    let seqs = report['seqs']
    for (let point_id in seqs) {
      //seq_text.push(point_id + ' :' + seqs[point_id]);
      seq_text.push(seqs[point_id]);
    }
    SeqsRender = (props) => {
      return (
        <ol>
          {seq_text.map((item, index) => {
            return (
              <ol class='roi-select-seq-strings-ol'>{item}</ol>
            );
          })}
        </ol>
      );
    }
    /*
    <p>{seq_text.map(index => {
            return (
              { seq_text[index]} < br >
      );
          })}</p>
          */
  }

  // Construct tab panes
  function RenderPanes() {
    return (
      [
        {
          // CDR3 report
          menuItem: { key: 'cdr3-length', content: 'CDR3 Length' },
          pane:
            <Tab.Pane key='cdr3-length-pane' className={'no-border'}>
              <ROICDR3Length
                report={report}
                section_name={'cdr3_lengths'}
                facetRowValues={facetRowValues}
                facetColValues={facetColValues}
                setAppStatus={setAppStatus}
              />
            </Tab.Pane>
        },
        {
          // V report
          menuItem: { key: 'v-report', content: 'V Distribution' },
          pane:
            <Tab.Pane key='v-report-pane' className={'no-border modal-roi-report'}>
              <ROIVDJ
                report={report}
                section_name={'vdist'}
                facetRowValues={facetRowValues}
                facetColValues={facetColValues}
                setAppStatus={setAppStatus}
              />
            </Tab.Pane>
        },
        {
          // J report
          menuItem: { key: 'j-report', content: 'J Distribution' },
          pane:
            <Tab.Pane key='j-report-pane' className={'no-border'}>
              <ROIVDJ
                report={report}
                section_name={'jdist'}
                facetRowValues={facetRowValues}
                facetColValues={facetColValues}
                setAppStatus={setAppStatus}
              />
            </Tab.Pane>
        },
        {
          // Report Information
          menuItem: { key: 'report-info', content: 'Stats' },
          pane:
            <Tab.Pane key='report-info-pane' className={'no-border'}>
              <ROIReportInformation
                report={report}
                section_name={'report_info'}
                setAppStatus={setAppStatus}
              />
            </Tab.Pane>
        },
        {
          // Sequences
          menuItem: { key: 'sequences', content: 'Sequences' },
          pane:
            <Tab.Pane key='sequences-pane' className={'no-border'}>
              <SeqsRender />
            </Tab.Pane>
        },
        {
          // Logo
          menuItem: { key: 'seqlogo', content: 'Logos' },
          pane:
            <Tab.Pane key='seqlogo-pane' className={'no-border'}>
              <LogoRender />
            </Tab.Pane>
        },
        {
          // System / debugging information
          menuItem: { key: 'system-info', content: 'System' },
          pane:
            <Tab.Pane key='system-info-pane' className={'no-border'}>
              <ROISystemInformation
                report={report}
                section_name={'_debug'}
                setAppStatus={setAppStatus}
              />
            </Tab.Pane>
        }
      ]
    );
  }

  function RenderLayout() {
    return (
      <Container fluid>
        <StatusMessage />
        <Tab menu={
          {
            secondary: true,
            pointing: true,
            color: 'blue'
          }
        }
          panes={RenderPanes()}
          renderActiveOnly={false}
          onTabChange={() => window.dispatchEvent(new Event('resize'))} // sizing issue workaround
        />
      </Container>
    );
  }
  return (
    <RenderLayout />
  );
}

export default React.memo(ROIReport)

