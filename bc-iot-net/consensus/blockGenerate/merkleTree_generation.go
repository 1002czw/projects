package blockGenerate

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"

)

type MerkleNode struct {
	Left *MerkleNode
	Right *MerkleNode
	Data []byte
}

type MerkleTree struct{
	RootNode *MerkleNode
}

func NewMerkleNode(left, right *MerkleNode, data []byte) *MerkleNode {
	node := MerkleNode{}
	// 创建存储明文信息的叶子节点
	if left == nil && right == nil {
		node.Data = data
		// 创建只有一个分支的MerkleNode
	} else if left != nil && right == nil {
		hash := sha256.Sum256(left.Data)
		node.Data = hash[:]
		// 创建有两个分支的MerkleNode
	} else {
		// slice = append(slice, anotherSlice...) 两个slice拼接在一起时要加...
		hash := sha256.Sum256(append(left.Data, right.Data...))
		node.Data = hash[:]
	}
	node.Left = left
	node.Right = right

	return &node
}

func NewMerkleTree(data [][]byte) *MerkleTree {
	var nodes []MerkleNode

	// 将所有数据构建为dataNode节点，接入node节点的左分支，并将node节存到nodes数组中
	for _, datum := range data {
		dataNode := NewMerkleNode(nil, nil, datum)
		node := NewMerkleNode(dataNode, nil, nil)
		nodes = append(nodes, *node)
	}

	for {
		var newLevel []MerkleNode

		// 根据当前层的节点，构造上一层
		// 当前层节点为奇数时
		if len(nodes)%2 == 1 {
			for j := 0; j < len(nodes)-1; j += 2 {
				node := NewMerkleNode(&nodes[j], &nodes[j+1], nil)
				newLevel = append(newLevel, *node)
			}
			node := NewMerkleNode(&nodes[len(nodes)-1], nil, nil)
			newLevel = append(newLevel, *node)
			// 当前层节点为偶数时
		} else {
			for j := 0; j < len(nodes); j += 2 {
				node := NewMerkleNode(&nodes[j], &nodes[j+1], nil)
				newLevel = append(newLevel, *node)
			}
		}

		// 更新层节点
		nodes = newLevel
		if len(nodes) == 1 {
			break
		}
	}
	mTree := MerkleTree{&nodes[0]}
	return &mTree
}

func PrintNode(node *MerkleNode) {
	fmt.Printf("%p\n", node)
	// 输出存储信息明文节点
	if node.Left != nil || node.Right != nil {
		fmt.Printf("left[%p], right[%p], data(%v)\n", node.Left, node.Right, hex.EncodeToString(node.Data))
		// 输出存储哈希值的节点
	} else if node.Left == nil || node.Right == nil {
		fmt.Printf("left[%p], right[%p], data(%v)\n", node.Left, node.Right, string(node.Data))
	}
}

type Transactions struct {
	Transaction [][]byte
}

func CreateMerkelTreeRoot(transactions Transactions) []byte{
	var tranHash [][]byte

	for _,tx:= range transactions.Transaction{

		tranHash = append(tranHash,tx)
	}

	mTree := NewMerkleTree(tranHash)

	Merkleroot :=  mTree.RootNode.Data
	return Merkleroot
}