package main

import (
	"bytes"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha1"
	"crypto/x509"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"github.com/hyperledger/fabric/core/chaincode/shim"
	"github.com/hyperledger/fabric/core/chaincode/shim/ext/entities"
	pb "github.com/hyperledger/fabric/protos/peer"
	"github.com/pkg/errors"
	"time"
)

const DOC_TYPE = "iotObj"

func (t *Iotchaincode) registerInfo(stub shim.ChaincodeStubInterface, args []string) pb.Response {

	var iot Iot
	//encKey := keys["ENCKEY"]
	//iv := keys["IV"]
	//fmt.Println(encKey,IV)

	err := json.Unmarshal([]byte(args[0]), &iot)
	if err != nil {
		fmt.Printf("function: registerInfo, Unmarshal errors:%s", err)
		return shim.Error("error")
	}

	// 查重: 身份证号码必须唯一
	_, err = stub.GetState(iot.EntityID)
	if err != nil {
		return shim.Error("the Iot ID has already been exist")
	}
	_, bl := putIotInfo(stub, iot)
	if !bl {
		fmt.Printf("function: rputIotInfo, error")
		return shim.Error("error")
	}

	err = stub.SetEvent(args[1], []byte{})
	if err != nil {
		return shim.Error(err.Error())
	}

	return shim.Success([]byte("register Information successful"))
}

func (t *Iotchaincode) deleteInfo(stub shim.ChaincodeStubInterface, args []string) pb.Response {
	iotName := args[0]
	err := stub.DelState(iotName)
	if err != nil {
		return shim.Error("删除信息时发生错误")
	}

	err = stub.SetEvent(args[1], []byte{})
	if err != nil {
		return shim.Error(err.Error())
	}

	return shim.Success([]byte("信息删除成功"))
}

func (t *Iotchaincode) queryInfo(stub shim.ChaincodeStubInterface, args []string) pb.Response {

	//get information by the key——entityID
	b, err := stub.GetState(args[0])
	if err != nil {
		fmt.Printf("query Information for %s is error: %s", args[0], err)
		return shim.Error("error")
	} else if b == nil {
		fmt.Printf("query Information, Iot does not existfor %s ", args[0])
		return shim.Error("error")
	}
	// 返回结果
	return shim.Success(b)

}

func (t *Iotchaincode) updateInfo(stub shim.ChaincodeStubInterface, args []string) pb.Response {
	var result, info Iot
	//encKey := keys["ENCKEY"]
	//IV := keys["IV"]

	err := json.Unmarshal([]byte(args[0]), &info)
	if err != nil {
		fmt.Printf("Function: updateInfo, unmarshal failed:%s", err)
		return shim.Error("error")
	}
	b, err := stub.GetState(info.EntityID)
	if err != nil {
		return shim.Error("updateInfo error")
	}
	if b == nil {
		return shim.Error("function updateInfo:the iot infomation does not exist")
	}
	// 对查询到的状态进行反序列化
	err = json.Unmarshal(b, &result)
	if err != nil {
		return shim.Error("function updateInfo:unmarshal error")
	}
	result.EntityID = info.EntityID
	result.Param = info.Param
	result.Historys = info.Historys

	_, bl := putIotInfo(stub, result)
	if !bl {
		fmt.Printf("updateInfo,when save information meet error")
		return shim.Error("error")
	}
	err = stub.SetEvent(args[1], []byte{})
	if err != nil {
		return shim.Error(err.Error())
	}

	return shim.Success([]byte("update iot information sucessful"))
}

func (t *Iotchaincode) queryInfoByDomain(stub shim.ChaincodeStubInterface, args []string) pb.Response {

	domainname := args[0]

	queryString := fmt.Sprintf("{\"selector\":{\"docType\":\"iotObj\",\"DomainName\":\"%s\"}}", domainname)

	queryResults, err := getQueryResultForQueryString(stub, queryString)
	if err != nil {
		return shim.Error(err.Error())
	}
	if queryResults == nil {
		return shim.Error("根据指定的Domain没有查询到相关的信息")
	}
	return shim.Success(queryResults)
}

func (t *Iotchaincode) getHistorybyEntity(stub shim.ChaincodeStubInterface, args []string) pb.Response {

	b, err := stub.GetState(args[0])
	if err != nil {
		fmt.Printf("function:getHistorybyEntity:get information failed:%s", err)
		return shim.Error("error")
	}

	if b == nil {
		fmt.Printf("function:getHistorybyEntity:information does not exist")
		return shim.Error("error")
	}

	// 对查询到的状态进行反序列化
	var iotinfo Iot
	err = json.Unmarshal(b, &iotinfo)
	if err != nil {
		fmt.Printf("function:getHistorybyEntity:unmarshal failed:%s", err)
		return shim.Error("error")
	}

	// 获取历史变更数据
	iterator, err := stub.GetHistoryForKey(iotinfo.EntityID)
	if err != nil {
		fmt.Printf("function:getHistorybyEntity:get history information by EntityID failed:%s", err)
		return shim.Error("error")
	}
	defer iterator.Close()

	// 迭代处理
	var historys []HistoryItem
	var hisIot Iot
	for iterator.HasNext() {
		hisData, err := iterator.Next()
		if err != nil {
			fmt.Printf("function:getHistorybyEntity:get hisIot history information failed:%s", err)
			return shim.Error("error")
		}

		var historyItem HistoryItem
		historyItem.TxId = hisData.TxId
		err = json.Unmarshal(hisData.Value, &hisIot)
		if err != nil {
			fmt.Printf("function:getHistorybyEntity:unmarshal failed:%s", err)
			return shim.Error("error")
		}

		if hisData.Value == nil {
			var empty Iot
			historyItem.Iot = empty
		} else {
			historyItem.Iot = hisIot
		}

		historys = append(historys, historyItem)

	}

	iotinfo.Historys = historys

	// 返回
	result, err := json.Marshal(iotinfo)
	if err != nil {
		fmt.Printf("function:getHistorybyEntity:marshal iot information failed:%s", err)
		return shim.Error("error")
	}
	return shim.Success(result)
}

func (t *Iotchaincode) payMoneyforKey(stub shim.ChaincodeStubInterface, args []string) pb.Response {
	var A, B string    // Entities
	var Aval, Bval int // Asset holdings
	var X int          // Transaction value
	var err error
	var resultA, resultB Iot

	if len(args) != 3 {
		return shim.Error("Incorrect number of arguments. Expecting 3")
	}

	A = args[0]
	B = args[1]

	// Get the state from the ledger
	re1, err := stub.GetState(A)
	if re1 == nil {
		return shim.Error("Entity not found")
	}
	if err != nil {
		return shim.Error("Failed to get state")
	}
	err = json.Unmarshal(re1, &resultA)
	Aval = resultA.Account
	//Aval, _ = strconv.Atoi(string(Avalbytes))

	re2, err := stub.GetState(B)
	if err != nil {
		return shim.Error("Failed to get state")
	}
	if re2 == nil {
		return shim.Error("Entity not found")
	}
	err = json.Unmarshal(re2, &resultB)
	Bval = resultB.Account
	//Bval, _ = strconv.Atoi(string(Bvalbytes))

	// Perform the execution
	X = resultB.Price
	if err != nil {
		return shim.Error("Invalid transaction amount, expecting a integer value")
	}
	Aval = Aval - X
	Bval = Bval + X
	//fmt.Printf("Aval = %d, Bval = %d\n", Aval, Bval)
	resultA.Account = Aval
	resultB.Account = Bval
	resultB.WhoCanSee = append(resultB.WhoCanSee, resultA.EntityID)
	_, bA := putIotInfo(stub, resultA)
	_, bB := putIotInfo(stub, resultB)
	if bA && bB {
		//Dec := RSAEncrypt([]byte("001001"),resultB.Publickey)
		//stub.PutState(resultA.Publickey,dec)
	} else {
		fmt.Printf("function: getDeckey, error")
		return shim.Error("error")
	}
	err = stub.SetEvent(args[2], []byte{})
	if err != nil {
		return shim.Error(err.Error())
	}
	fmt.Println("submitDecInfoByMoney success")
	return shim.Success(nil)
}

func (t *Iotchaincode) submitKey(stub shim.ChaincodeStubInterface, args []string) pb.Response {

	keys := args[1]
	var userList, usr Iot
	b, err := stub.GetState(args[0])
	if err != nil {
		fmt.Printf("submitKey query Information for %s is error: %s", args[0], err)
		return shim.Error("error")
	} else if b == nil {
		fmt.Printf("submitKey query Information, Iot does not existfor %s ", args[0])
		return shim.Error("error")
	}
	json.Unmarshal(b, &userList)
	list := userList.WhoCanSee
	for i := 0; i < len(list); i++ {
		a, err := stub.GetState(list[i])
		if err != nil {
			fmt.Println("error")
		}
		json.Unmarshal(a, &usr)
		dec := RSAEncrypt([]byte(keys), usr.Publickey)
		var submitkey Encrykeys
		submitkey.GenerateTime = time.Now()
		submitkey.Key = dec
		submitkey.DurationTime = 6
		d, err := json.Marshal(submitkey)
		stub.PutState(getSHA1(usr.Publickey), d)
	}
	err = stub.SetEvent(args[2], []byte{})
	if err != nil {
		return shim.Error(err.Error())
	}
	fmt.Println("submitDecInfoByMoney success")
	return shim.Success(nil)
}
func (t *Iotchaincode) computerAvgModel(stub shim.ChaincodeStubInterface, args []string) pb.Response {
	var err error
	var resultA, resultB, resultC Iot
	r0, err := stub.GetState(args[0])
	if r0 == nil {
		return shim.Error("Entity not found")
	}
	if err != nil {
		return shim.Error("Failed to get state")
	}
	r1, err := stub.GetState(args[1])
	if r1 == nil {
		return shim.Error("Entity not found")
	}
	if err != nil {
		return shim.Error("Failed to get state")
	}
	r2, err := stub.GetState(args[2])
	if r2 == nil {
		return shim.Error("Entity not found")
	}
	if err != nil {
		return shim.Error("Failed to get state")
	}

	err = json.Unmarshal(r0, &resultA)
	if err != nil {
		return shim.Error("Failed unmarshal r0")
	}
	err = json.Unmarshal(r1, &resultB)
	if err != nil {
		return shim.Error("Failed unmarshal r1")
	}
	err = json.Unmarshal(r2, &resultC)
	if err != nil {
		return shim.Error("Failed unmarshal r2")
	}

	a1 := resultA.Param[0].Param_W
	a2 := resultA.Param[1].Param_b
	a3 := resultA.Param[2].Param_m[0]
	b1 := resultB.Param[0].Param_W
	b2 := resultB.Param[1].Param_b
	b3 := resultB.Param[2].Param_m[0]
	c1 := resultC.Param[0].Param_W
	c2 := resultC.Param[1].Param_b
	c3 := resultC.Param[2].Param_m[0]
	resultSumW := [5][2]float64{}
	resultSumb := [2]float64{}
	total_size := a3 + b3 + c3

	for i := 0; i < len(a1); i++ {
		for j := 0; j < len(a1[0]); j++ {
			resultSumW[i][j] = a1[i][j] + b1[i][j] + c1[i][j]
		}
	}
	for i := 0; i < len(a2); i++ {
		resultSumb[i] = a2[i] + b2[i] + c2[i]
	}
	fmt.Println(total_size)

	for i := 0; i < len(resultSumW); i++ {
		for j := 0; j < len(resultSumW[0]); j++ {
			resultSumW[i][j] = resultSumW[i][j] / total_size
		}
	}
	for i := 0; i < len(resultSumb); i++ {
		resultSumb[i] = resultSumb[i] / total_size
	}

	totalCost := (resultA.Param[2].Param_m[1] + resultB.Param[2].Param_m[1] + resultC.Param[2].Param_m[1]) / 3.0
	var res ComputerRes
	res.Total_grad_W = resultSumW
	res.Total_grad_b = resultSumb
	res.Total_cost = [2]float64{total_size, totalCost}
	d, err := json.Marshal(res)
	if err != nil {
		fmt.Printf("Marshal is error: %s", err)
		return shim.Error("error")
	}
	stub.PutState("Fed_Block", d)

	n := len(args)
	err = stub.SetEvent(args[n-1], []byte{})
	if err != nil {
		return shim.Error(err.Error())
	}
	fmt.Println("computing success")
	return shim.Success(d)
}

func RSAEncrypt(data []byte, publicKey string) []byte {
	//从数据中查找到下一个PEM格式的块
	var err error
	block, _ := pem.Decode([]byte(publicKey))
	if block == nil {
		fmt.Println("publicKey decode error")
		return nil
	}
	//解析一个DER编码的公钥
	pubInterface, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		fmt.Println("ParsePKIXPublicKey解析公钥 error")
	}
	publicKey1 := pubInterface.(*rsa.PublicKey)
	//公钥加密
	result, _ := rsa.EncryptPKCS1v15(rand.Reader, publicKey1, data)
	return result
}

//function
func putIotInfo(stub shim.ChaincodeStubInterface, iot Iot) ([]byte, bool) {
	iot.ObjectType = DOC_TYPE

	b, err := json.Marshal(iot)
	if err != nil {
		fmt.Printf("function: putIotInfo, marshal iot error: %s", err)
		return nil, false
	}

	//save the iot information
	err = stub.PutState(iot.EntityID, b)
	if err != nil {
		fmt.Printf("function: putIotInfo, putstate the iot information error: %s", err)
		return nil, false
	}
	return nil, true
}

func getStateAndDecrypt(stub shim.ChaincodeStubInterface, ent entities.Encrypter, key string) ([]byte, error) {
	// at first we retrieve the ciphertext from the ledger
	ciphertext, err := stub.GetState(key)
	if err != nil {
		return nil, err
	}

	// GetState will return a nil slice if the key does not exist.
	// Note that the chaincode logic may want to distinguish between
	// nil slice (key doesn't exist in state db) and empty slice
	// (key found in state db but value is empty). We do not
	// distinguish the case here
	if len(ciphertext) == 0 {
		return nil, errors.New("no ciphertext to decrypt")
	}

	return ent.Decrypt(ciphertext)
}

func getQueryResultForQueryString(stub shim.ChaincodeStubInterface, queryString string) ([]byte, error) {

	resultsIterator, err := stub.GetQueryResult(queryString)
	if err != nil {
		return nil, err
	}
	defer resultsIterator.Close()
	// buffer is a JSON array containing QueryRecords
	var buffer bytes.Buffer

	bArrayMemberAlreadyWritten := false
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}
		// Add a comma before array members, suppress it for the first array member
		if bArrayMemberAlreadyWritten == true {
			buffer.WriteString(",")
		}

		// Record is a JSON object, so we write as-is
		buffer.WriteString(string(queryResponse.Value))
		bArrayMemberAlreadyWritten = true
	}

	fmt.Printf("- getQueryResultForQueryString queryResult:\n%s\n", buffer.String())

	return buffer.Bytes(), nil
	/*
		buffer, err := constructQueryResponseFromIterator(resultsIterator)
		if err != nil {
			return nil, err
		}

		fmt.Printf("- getQueryResultForQueryString queryResult:\n%s\n", buffer.String())

		return buffer.Bytes(), nil

	*/
}
func (t *Iotchaincode) getInfoByRange(stub shim.ChaincodeStubInterface, args []string) pb.Response {
	return shim.Success(nil)
}

func constructQueryResponseFromIterator(resultsIterator shim.StateQueryIteratorInterface) (*bytes.Buffer, error) {
	// buffer is a JSON array containing QueryResults
	var buffer bytes.Buffer
	buffer.WriteString("[")

	bArrayMemberAlreadyWritten := false
	for resultsIterator.HasNext() {
		queryResponse, err := resultsIterator.Next()
		if err != nil {
			return nil, err
		}
		// Add a comma before array members, suppress it for the first array member
		if bArrayMemberAlreadyWritten == true {
			buffer.WriteString(",")
		}
		buffer.WriteString("{\"Key\":")
		buffer.WriteString("\"")
		buffer.WriteString(queryResponse.Key)
		buffer.WriteString("\"")

		buffer.WriteString(", \"Record\":")
		// Record is a JSON object, so we write as-is
		buffer.WriteString(string(queryResponse.Value))
		buffer.WriteString("}")
		bArrayMemberAlreadyWritten = true
	}
	buffer.WriteString("]")
	return &buffer, nil

}

func getSHA1(str string) string {

	myhash := sha1.New()
	//3.将文件数据拷贝给Hash对象
	myhash.Write([]byte(str))
	//4.计算文件的哈希值
	temp := myhash.Sum(nil)
	//5.转换数据格式
	//fmt.Println(temp)
	result := hex.EncodeToString(temp)
	res := result[:10]
	return res
}
