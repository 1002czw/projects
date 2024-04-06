package blockGenerate

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"github.com/iot-data-sharing-project/application/cli_service"
	"strconv"
	"sync"
)

var mutex = &sync.Mutex{} //用sync防止同一时间产生多个区块

func CalculateHash(block cli_service.Block) string {
	record := strconv.Itoa(block.Index) + block.Timestamp + block.Data + block.PrevHash + block.Nonce
	h := sha256.New()
	h.Write([]byte(record))
	hashed := h.Sum(nil)
	return hex.EncodeToString(hashed)
}

func IsBlockValid(newBlock, oldBlock cli_service.Block, checkChannel chan bool) bool {
	defer wp.Done()
	if oldBlock.Index+1 != newBlock.Index {
		return false //确认Index的增长正确
	} //双重否定（笑）
	if oldBlock.Hash != newBlock.PrevHash {
		return false //确认PrevHash与前一个块的Hash相同
	}
	if CalculateHash(newBlock) != newBlock.Hash {
		//在当前块上 calculateHash 再次运行该函数来检查当前块的Hash
		return false
	}
	checkChannel <- true
	return true
}

var wp sync.WaitGroup

func ChechBlock(cli_num int, winner cli_service.Consensus_competing_result, oldBlock cli_service.Block, checkChannel chan bool) bool {
	for cli := 1; cli <= cli_num; cli++ {
		client := fmt.Sprintf("Client_%d", cli)
		if client != winner.Name {

			go IsBlockValid(winner.Block, oldBlock, checkChannel)
			wp.Add(1)
		}
	}
	wp.Wait()

	true_num := 0
	false_num := 0
	for num := 1; num <= cli_num; num++ {
		select {
		case check_r := <-checkChannel:
			if check_r == true {
				true_num = true_num + 1
			} else {
				false_num = false_num + 1
			}
		default:
			fmt.Printf("Read check results finished----->")
		}
	}

	if true_num/(true_num+false_num) > 2/3 {
		return true
	} else {
		return false
	}
}
