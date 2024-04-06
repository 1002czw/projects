#! /bin/bash
echo "定义名称..."
cli0=cli_org1_peer0
cli1=cli_org1_peer1
cli2=cli_org2_peer0
cli3=cli_org2_peer1
peer_ca_adreess=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/iot.com/orderers/orderer.iot.com/msp/tlscacerts/tlsca.iot.com-cert.pem
orderer_adreess=orderer.iot.com:7050
channelblock_adreess=/opt/gopath/src/github.com/hyperledger/fabric/peer/iotchannel.block
echo "完成定义..."
echo "进入容器cli_org1_peer0"
  docker exec $cli0 scripts/script.sh $CHANNEL_NAME $CLI_DELAY $LANGUAGE $CLI_TIMEOUT $VERBOSE
  if [ $? -ne 0 ]; then
    echo "ERROR !!!! Test failed"
    exit 1
  fi

