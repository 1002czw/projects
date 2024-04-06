package cli_service

import (
	"encoding/json"
	"fmt"
	"github.com/hyperledger/fabric-sdk-go/pkg/client/channel"
)

func (t *ServiceSetup) AppendBlockIOT(bc BlockAppend) (string, error) {
	eventID := "eventAppendBlock"
	reg, notifier := regitserEvent(t.Client, t.ChaincodeID, eventID)
	defer t.Client.UnregisterChaincodeEvent(reg)

	// 将edu对象序列化成为字节数组
	b, err := json.Marshal(bc)
	if err != nil {
		return "", fmt.Errorf("AppendBlockIOT marshal error")
	}

	req := channel.Request{ChaincodeID: t.ChaincodeID, Fcn: "generateBlock", Args: [][]byte{b, []byte(eventID)}} //

	respone, err := t.Client.Execute(req) //channel.WithTargetEndpoints("peer0.org1.iot.com")
	if err != nil {
		return " ", err
	}

	err = eventResult(notifier, eventID)
	if err != nil {
		return "", err
	}

	return string(respone.TransactionID), nil //

}

func (t *ServiceSetup) RegisterIOT(iot Iot) (string, error) {
	eventID := "eventRegister"
	reg, notifier := regitserEvent(t.Client, t.ChaincodeID, eventID)
	defer t.Client.UnregisterChaincodeEvent(reg)

	// 将edu对象序列化成为字节数组
	b, err := json.Marshal(iot)
	if err != nil {
		return "", fmt.Errorf("指定的iot对象序列化时发生错误")
	}

	req := channel.Request{ChaincodeID: t.ChaincodeID, Fcn: "registerInfo", Args: [][]byte{b, []byte(eventID)}} //

	respone, err := t.Client.Execute(req) //channel.WithTargetEndpoints("peer0.org1.iot.com")
	if err != nil {
		return " ", err
	}

	err = eventResult(notifier, eventID)
	if err != nil {
		return "", err
	}

	return string(respone.TransactionID), nil //

}

func (t *ServiceSetup) UpdateContriInfoIOT(contri EvalCli) (string, error) {
	eventID := "eventUpdateContriInfo"
	reg, notifier := regitserEvent(t.Client, t.ChaincodeID, eventID)
	defer t.Client.UnregisterChaincodeEvent(reg)

	// 将edu对象序列化成为字节数组
	b, err := json.Marshal(contri)
	if err != nil {
		return "", fmt.Errorf("指定的UpdateContriInfo对象序列化时发生错误")
	}

	req := channel.Request{ChaincodeID: t.ChaincodeID, Fcn: "updateContriInfo", Args: [][]byte{b, []byte(eventID)}} //

	respone, err := t.Client.Execute(req) //channel.WithTargetEndpoints("peer0.org1.iot.com")
	if err != nil {
		return " ", err
	}

	err = eventResult(notifier, eventID)
	if err != nil {
		return "", err
	}

	return string(respone.TransactionID), nil //

}



func (t *ServiceSetup) UploadContriInfoIOT(contri EvalCli) (string, error) {
	eventID := "eventUploadContriInfo"
	reg, notifier := regitserEvent(t.Client, t.ChaincodeID, eventID)
	defer t.Client.UnregisterChaincodeEvent(reg)

	// 将edu对象序列化成为字节数组
	b, err := json.Marshal(contri)
	if err != nil {
		return "", fmt.Errorf("指定的UploadContriInfo对象序列化时发生错误")
	}

	req := channel.Request{ChaincodeID: t.ChaincodeID, Fcn: "uploadContriInfo", Args: [][]byte{b, []byte(eventID)}} //

	respone, err := t.Client.Execute(req) //channel.WithTargetEndpoints("peer0.org1.iot.com")
	if err != nil {
		return " ", err
	}

	err = eventResult(notifier, eventID)
	if err != nil {
		return "", err
	}

	return string(respone.TransactionID), nil //

}

func (t *ServiceSetup) UploadConsensusCommitteeIOT(com Consensus_committee) (string, error) {
	eventID := "eventConsensusCommittee"
	reg, notifier := regitserEvent(t.Client, t.ChaincodeID, eventID)
	defer t.Client.UnregisterChaincodeEvent(reg)

	// 将edu对象序列化成为字节数组
	b, err := json.Marshal(com)
	if err != nil {
		return "", fmt.Errorf("指定的ConsensusCommittee对象序列化时发生错误")
	}

	req := channel.Request{ChaincodeID: t.ChaincodeID, Fcn: "uploadConsensusMembers", Args: [][]byte{b, []byte(eventID)}} //

	respone, err := t.Client.Execute(req) //channel.WithTargetEndpoints("peer0.org1.iot.com")
	if err != nil {
		return " ", err
	}

	err = eventResult(notifier, eventID)
	if err != nil {
		return "", err
	}

	return string(respone.TransactionID), nil //

}

//release Task
func (t *ServiceSetup) ReleaseTaskIOT(model ModelInit) (string, error) {
	eventID := "eventTaskRelease"
	reg, notifier := regitserEvent(t.Client, t.ChaincodeID, eventID)
	defer t.Client.UnregisterChaincodeEvent(reg)

	// 将edu对象序列化成为字节数组
	b, err := json.Marshal(model)
	if err != nil {
		return "", fmt.Errorf("指定的ModelInit对象序列化时发生错误")
	}

	req := channel.Request{ChaincodeID: t.ChaincodeID, Fcn: "releaseTaskInfo", Args: [][]byte{b, []byte(eventID)}} //

	respone, err := t.Client.Execute(req) //channel.WithTargetEndpoints("peer0.org1.iot.com")
	if err != nil {
		return " ", err
	}

	err = eventResult(notifier, eventID)
	if err != nil {
		return "", err
	}

	return string(respone.TransactionID), nil //

}

//delete
func (t *ServiceSetup) DelIOT(entityID string) (string, error) {

	eventID := "eventDelIot"
	reg, notifier := regitserEvent(t.Client, t.ChaincodeID, eventID)
	defer t.Client.UnregisterChaincodeEvent(reg)

	req := channel.Request{ChaincodeID: t.ChaincodeID, Fcn: "deleteInfo", Args: [][]byte{[]byte(entityID), []byte(eventID)}}
	respone, err := t.Client.Execute(req)
	if err != nil {
		return "", err
	}

	err = eventResult(notifier, eventID)
	if err != nil {
		return "", err
	}

	return string(respone.TransactionID), nil

}

func (t *ServiceSetup) FindInfoByEntityID(entityID string) ([]byte, error) {
	eventID := "eventQueryIOT"
	reg, _ := regitserEvent(t.Client, t.ChaincodeID, eventID)
	defer t.Client.UnregisterChaincodeEvent(reg)
	req := channel.Request{ChaincodeID: t.ChaincodeID, Fcn: "queryInfo", Args: [][]byte{[]byte(entityID), []byte(eventID)}}
	respone, err := t.Client.Query(req)
	if err != nil {
		return []byte{0x00}, err
	}
	fmt.Println("query successful")
	return respone.Payload, nil
}

func (t *ServiceSetup) UpdateIOT(iot Iot) (string, error) {

	eventID := "eventUpdateIOT"
	reg, notifier := regitserEvent(t.Client, t.ChaincodeID, eventID)
	defer t.Client.UnregisterChaincodeEvent(reg)

	// 将edu对象序列化成为字节数组
	b, err := json.Marshal(iot)
	if err != nil {
		return "", fmt.Errorf("指定的edu对象序列化时发生错误")
	}

	req := channel.Request{ChaincodeID: t.ChaincodeID, Fcn: "updateInfo", Args: [][]byte{b, []byte(eventID)}}
	respone, err := t.Client.Execute(req)
	if err != nil {
		return "", err
	}

	err = eventResult(notifier, eventID)
	if err != nil {
		return "", err
	}

	return string(respone.TransactionID), nil
}

func (t *ServiceSetup) FindInfoByDomain(name string) ([]byte, error) {
	eventID := "eventQueryIOTByDomain"
	reg, _ := regitserEvent(t.Client, t.ChaincodeID, eventID)
	defer t.Client.UnregisterChaincodeEvent(reg)
	req := channel.Request{ChaincodeID: t.ChaincodeID, Fcn: "queryInfoByDomain", Args: [][]byte{[]byte(name), []byte(eventID)}}
	respone, err := t.Client.Query(req)
	if err != nil {
		return []byte{0x00}, err
	}
	return respone.Payload, nil
}

func (t *ServiceSetup) GetHistory(entity string) ([]byte, error) {
	eventID := "eventGetHistory"
	reg, _ := regitserEvent(t.Client, t.ChaincodeID, eventID)
	defer t.Client.UnregisterChaincodeEvent(reg)
	req := channel.Request{ChaincodeID: t.ChaincodeID, Fcn: "getHistorybyEntity", Args: [][]byte{[]byte(entity), []byte(eventID)}}
	respone, err := t.Client.Execute(req)
	if err != nil {
		return []byte{0x00}, err
	}

	return respone.Payload, nil
}
func (t *ServiceSetup) PayMoneyforKeyIOT(a, b string) (string, error) {

	eventID := "eventPayMoneyforKeyIOT"
	reg, notifier := regitserEvent(t.Client, t.ChaincodeID, eventID)
	defer t.Client.UnregisterChaincodeEvent(reg)

	req := channel.Request{ChaincodeID: t.ChaincodeID, Fcn: "payMoneyforKey", Args: [][]byte{[]byte(a), []byte(b), []byte(eventID)}} //

	respone, err := t.Client.Execute(req) //channel.WithTargetEndpoints("peer0.org1.iot.com")
	if err != nil {
		return " ", err
	}

	err = eventResult(notifier, eventID)
	if err != nil {
		return "", err
	}

	return string(respone.TransactionID), nil //

}

func (t *ServiceSetup) SubmitKeyIOT(a, b string) (string, error) {

	eventID := "eventSubmitKeyIOT"
	reg, notifier := regitserEvent(t.Client, t.ChaincodeID, eventID)
	defer t.Client.UnregisterChaincodeEvent(reg)

	req := channel.Request{ChaincodeID: t.ChaincodeID, Fcn: "submitKey", Args: [][]byte{[]byte(a), []byte(b), []byte(eventID)}} //

	respone, err := t.Client.Execute(req) //channel.WithTargetEndpoints("peer0.org1.iot.com")
	if err != nil {
		return " ", err
	}

	err = eventResult(notifier, eventID)
	if err != nil {
		return "", err
	}

	return string(respone.TransactionID), nil //

}

func (t *ServiceSetup) ComputerAvgModelIOT(a, b, c string) ([]byte, error) {

	eventID := "eventComputerAvgModelIOT"
	reg, notifier := regitserEvent(t.Client, t.ChaincodeID, eventID)
	defer t.Client.UnregisterChaincodeEvent(reg)

	req := channel.Request{ChaincodeID: t.ChaincodeID, Fcn: "computerAvgModel", Args: [][]byte{[]byte(a), []byte(b), []byte(c), []byte(eventID)}} //

	respone, err := t.Client.Execute(req) //channel.WithTargetEndpoints("peer0.org1.iot.com")
	if err != nil {
		return []byte("error"), err
	}

	err = eventResult(notifier, eventID)
	if err != nil {
		return []byte("error"), err
	}

	return respone.Payload, nil //

}
