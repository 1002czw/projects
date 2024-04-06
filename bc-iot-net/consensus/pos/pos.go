package pos


import (
"crypto/sha256"
"encoding/hex"
"fmt" /* Package os为操作系统功能提供了一个平台无关的接口。虽然错误处理类似于 Go，但设计类似                                                   Unix，失败的调用返回类型错误的值而不是错误号  */
"github.com/iot-data-sharing-project/application/cli_service"
"github.com/iot-data-sharing-project/bc-iot-net/consensus/blockGenerate"
"runtime"
"strconv" //包strconv实现了对基本数据类型的字符串表示的转换
"strings" //打包字符串实现简单的函数来操纵 UTF-8 编码的字符串
"sync"
"time" // 打包时间提供了测量和显示时间的功能。日历计算总是假定公历，没有闰秒
"os/exec"
"math/rand"
)



type Pos_result struct {
	Name      string            `json:"name"`
	Judge_out bool              `json:"judge_Out"`
	Time      float64           `json:"time"`
	Nonce     string            `json:"nonce"`
	Block     cli_service.Block `json:"block"`
}

const difficulty = 2 //difficulty

func PoS(cli_num int, transactions [][]byte, train_label bool) cli_service.Consensus_competing_result {
	var wp sync.WaitGroup
	stopChannel := make(chan cli_service.Consensus_competing_result, cli_num)
	for i := 1; i <= cli_num; i++ {
		//defer wp.Done()
		if train_label{
			cli_win := fmt.Sprintf("client_%d", i)
			command_Aggregation := fmt.Sprintf("~/go/src/github.com/iot-data-sharing-project/application/users/%s/aggregation_Global.sh %s ", cli_win, cli_win)
			cmd := exec.Command("/bin/bash", "-c", command_Aggregation)
			_, err_simu := cmd.Output()
			if err_simu != nil {
				fmt.Printf("Execute Shell:%s failed with error:%s \n", command_Aggregation, err_simu.Error())
				break
			}
		}
		tree := blockGenerate.NewMerkleTree(transactions)
		Merkleroot_hash := hex.EncodeToString(tree.RootNode.Data)
		wp.Add(1)
		go GenerateBlock(cli_service.Blockchain[len(cli_service.Blockchain)-1], Merkleroot_hash, stopChannel, i, &wp )

	}
	wp.Wait()
	win_result := <-stopChannel
	//
	return win_result

}
func GenerateBlock(oldBlock cli_service.Block, Data string, stopChannel chan cli_service.Consensus_competing_result, cli_i int, wp *sync.WaitGroup)  { //定义函数generateBlock
	defer wp.Done()
	var newBlock cli_service.Block //新区块
	t := time.Now()
	newBlock.Index = oldBlock.Index + 1 //区块的增加，index也加一
	newBlock.Timestamp = t.String()     //时间戳
	newBlock.Data = Data
	newBlock.PrevHash = oldBlock.Hash //新区块的PrevHash存储上一个区块的Hash
	newBlock.Height = oldBlock.Height + 1
	newBlock.Difficulty = int(difficulty* rand.Float64()+0.5)
	newBlock.ConsensusType = "pos"
	for i := 0; ; i++ { //通过循环改变 Nonce
		l := len(stopChannel)
		if l > 0 {
			runtime.Goexit()
		}
		hex := fmt.Sprintf("%x", i)
		newBlock.Nonce = hex //选出符合难度系数的Nonce
		if !IsHashValid(CalculateHash(newBlock), newBlock.Difficulty) {
			//判断Hash的0的个数，是否与难度系数一致
			fmt.Printf("%s cli %d do more work!\n", CalculateHash(newBlock), cli_i) //挖矿中
			time.Sleep(time.Second)
			continue
		} else {
			fmt.Printf("%s cli %d work done\n", CalculateHash(newBlock), cli_i) //挖矿成功
			newBlock.Hash = CalculateHash(newBlock)
			newBlock.Nonce = hex
			inputInfo := cli_service.Consensus_competing_result{
				Name:      fmt.Sprintf("Client_%d", cli_i),
				Judge_out: true,
				Time:      time.Now(),
				Nonce:     hex,
				Block:     newBlock,
			}
			//fmt.Println(inputInfo)
			stopChannel <- inputInfo
			//fmt.Println(stopChannel)
			//r ,ok:= <- stopChannel
			//fmt.Printf("cli_%d, %s,%t\n",cli_i,r,ok)
			//time.Sleep(time.Second)

			fmt.Println(len(stopChannel))
			runtime.Goexit()
		}
	}

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
