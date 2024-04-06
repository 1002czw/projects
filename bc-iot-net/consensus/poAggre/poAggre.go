package poAggre

import (
	"encoding/hex"
	"fmt"
	"github.com/iot-data-sharing-project/application/cli_service"
	"github.com/iot-data-sharing-project/bc-iot-net/consensus/blockGenerate"
	"os/exec"
	"strings"
	"sync"
	"crypto/sha256"
	"strconv"
	"time"
)

func PoAggre(cli_num int, transactions [][]byte) cli_service.Consensus_competing_result{
	var wp sync.WaitGroup
	stopChannel := make(chan cli_service.Consensus_competing_result, cli_num)
	for i := 1; i <= cli_num; i++ {
		client := fmt.Sprintf("Client_%d", i)
		wp.Add(1)
		go aggregation(client, &wp, stopChannel, transactions)
	}
	wp.Wait()
	//check the aggregation result
	fmt.Println("check the aggregation result")
	time.Sleep(2*time.Second)
	fmt.Println("check the aggregation result finished")

	win_result := <- stopChannel
	return win_result

}

func aggregation(client string, wp *sync.WaitGroup, stopChannel chan cli_service.Consensus_competing_result, transactions [][]byte) {
	defer wp.Done()
	cli_win := strings.ToLower(client)
	command_Aggregation := fmt.Sprintf("~/go/src/github.com/iot-data-sharing-project/application/users/%s/aggregation_Global.sh %s ", cli_win, cli_win)
	cmd := exec.Command("/bin/bash", "-c", command_Aggregation)
	_, err_simu := cmd.Output()
	if err_simu != nil {
		fmt.Printf("Execute Shell:%s failed with error:%s \n", command_Aggregation, err_simu.Error())
		return
	}

	aggre_Finish_time:=time.Now()
	tree := blockGenerate.NewMerkleTree(transactions)
	Merkleroot_hash := hex.EncodeToString(tree.RootNode.Data)
	var newBlock cli_service.Block //新区块
	t := time.Now()
	oldBlock := cli_service.Blockchain[len(cli_service.Blockchain)-1]
	newBlock.Index = oldBlock.Index + 1 //区块的增加，index也加一
	newBlock.Timestamp = t.String()     //时间戳
	newBlock.Data = Merkleroot_hash
	newBlock.PrevHash = oldBlock.Hash //新区块的PrevHash存储上一个区块的Hash
	newBlock.Height = oldBlock.Height + 1
	newBlock.Difficulty = 0
	newBlock.Hash = CalculateHash(newBlock)
	newBlock.ConsensusType = "poAggre"
	inputInfo := cli_service.Consensus_competing_result{
		Name:         client,
		Judge_out:    true,
		Time:         aggre_Finish_time,
		Aggre_result: fmt.Sprintf("%s aggre result address",client),
		Block: newBlock,
	}
	stopChannel <- inputInfo

}

func CalculateHash(block cli_service.Block) string {
	record := strconv.Itoa(block.Index) + block.Timestamp + block.Data + block.PrevHash + block.Nonce
	h := sha256.New()
	h.Write([]byte(record))
	hashed := h.Sum(nil)
	return hex.EncodeToString(hashed)
}
