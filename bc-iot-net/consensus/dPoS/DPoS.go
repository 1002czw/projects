package dPoS

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt" /* Package os为操作系统功能提供了一个平台无关的接口。虽然错误处理类似于 Go，但设计类似                                                   Unix，失败的调用返回类型错误的值而不是错误号  */
	"github.com/iot-data-sharing-project/application/cli_service"
	"github.com/iot-data-sharing-project/bc-iot-net/consensus/blockGenerate"
	"strconv" //包strconv实现了对基本数据类型的字符串表示的转换
	"strings" //打包字符串实现简单的函数来操纵 UTF-8 编码的字符串
	"time" // 打包时间提供了测量和显示时间的功能。日历计算总是假定公历，没有闰秒
	"math/rand"
	"encoding/json"
)



type DPos_result struct {
	Name      string            `json:"name"`
	Judge_out bool              `json:"judge_Out"`
	Time      float64           `json:"time"`
	Nonce     string            `json:"nonce"`
	Block     cli_service.Block `json:"block"`
}
// 验证者列表




// 选择验证者的函数

const difficulty = 0 //difficulty

func DPoS( transactions [][]byte,serviceSetup1 cli_service.ServiceSetup, train_on_going bool) cli_service.Consensus_competing_result {
	// 验证者验证区块

	validators, err := serviceSetup1.FindInfoByEntityID("DPoS_Committee")
	if err != nil {
		fmt.Printf("FindInfoByEntityID(\"DPoS_Committee\") error: %s", err.Error())
	}
	var vali cli_service.Consensus_committee
	if err := json.Unmarshal(validators, &vali); err != nil {
		fmt.Printf("Failed to unmarshal query result, err :%s",err)
	}

	validator:= selectValidator(vali.Members)

	win_result := cli_service.Consensus_competing_result{}

	tree := blockGenerate.NewMerkleTree(transactions)
	Merkleroot_hash := hex.EncodeToString(tree.RootNode.Data)
	newBlock:=GenerateBlock(cli_service.Blockchain[len(cli_service.Blockchain)-1], validator,Merkleroot_hash)
	// 生成新的区块
	win_result.Name=validator
	win_result.Judge_out=true
	win_result.Time=time.Now()
	win_result.Block=newBlock
	//
	return win_result

}

func selectValidator(validators []cli_service.Consensus_member) string {
	// 设置随机种子
	rand.Seed(time.Now().UnixNano())

	// 随机选择一个验证者
	index := rand.Intn(len(validators))
	return validators[index].Name.(string)
}

func GenerateBlock(oldBlock cli_service.Block, validator, Data string)  cli_service.Block{ //定义函数generateBlock
	var newBlock cli_service.Block //新区块
	t := time.Now()
	newBlock.Index = oldBlock.Index + 1 //区块的增加，index也加一
	newBlock.Timestamp = t.String()     //时间戳
	newBlock.Data = Data
	newBlock.PrevHash = oldBlock.Hash //新区块的PrevHash存储上一个区块的Hash
	newBlock.Height = oldBlock.Height + 1
	newBlock.Difficulty = difficulty
	newBlock.ConsensusType = "dpos"
	newBlock.Hash = CalculateHash(newBlock)
	newBlock.Nonce = "0"
	newBlock.Hash = CalculateHash(newBlock)

	return newBlock

}

func IsHashValid(hash string, difficulty int) bool {

	prefix := strings.Repeat("0", difficulty)
	//复制 difficulty 个0，并返回新字符串，当 difficulty 为 4 ，则 prefix 为 0000
	return strings.HasPrefix(hash, prefix) // 判断字符串 hash 是否包含前缀 prefix
}

func CalculateHash(block cli_service.Block) string {
	record := strconv.Itoa(block.Index) + block.Timestamp + block.Data + block.PrevHash + block.Nonce
	h := sha256.New()
	h.Write([]byte(record))
	hashed := h.Sum(nil)
	return hex.EncodeToString(hashed)
}
