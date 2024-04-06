#！ /bin/bash
echo "##############################################################"
echo "######       START TO CONSTRUCT BLOCKCHAIN NETWORK      ######"
echo "##############################################################"
echo "                                                              "
echo "##############################################################"
echo "##########           orderers starting          ##############"
echo "##############################################################"
docker-compose -f ./iot/docker-orderer.yaml up -d
echo "                                                              "
echo "##############################################################"
echo "######  couch0 & org1-peer0 & cli_org1_peer0 starting   ######"
echo "##############################################################"
docker-compose -f ./iot/docker-org1-peer0.yaml up -d
echo "                                                              "
echo "##############################################################"
echo "##########        couch1 & org1-peer1 starting       #########"
echo "##############################################################"
docker-compose -f ./iot/docker-org1-peer1.yaml up -d
echo "                                                              "
echo "##############################################################"
echo "##########          cli_org1_peer1 starting         ##########"
echo "##############################################################"
docker-compose -f ./iot/docker-org1-peer1-cli.yaml up -d
echo "                                                              "
echo "##############################################################"
echo "######   couch2 & org2-peer0 & cli_org2_peer0 starting  ######"
echo "##############################################################"
docker-compose -f ./iot/docker-org2-peer0.yaml up -d
echo "                                                              "
echo "##############################################################"
echo "##########        couch3 & org2-peer1 starting       #########"
echo "##############################################################"
docker-compose -f ./iot/docker-org2-peer1.yaml up -d
echo "                                                              "
echo "##############################################################"
echo "##########          cli_org2_peer1 starting         ##########"
echo "##############################################################"
docker-compose -f ./iot/docker-org2-peer1-cli.yaml up -d
echo "                                                              "
echo "##############################################################"
echo "#IOT-Data-Sharing-Collaborative-Networks building successful!#"
echo "##############################################################"
./iot/scripts/first.sh

