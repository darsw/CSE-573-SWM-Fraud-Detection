CREATE CONSTRAINT ON (c:Customer) ASSERT c.id IS UNIQUE;
CREATE CONSTRAINT ON (m:Merchant) ASSERT m.id IS UNIQUE;

LOAD CSV WITH HEADERS FROM
"file:///Data/dataset.csv" AS line
WITH line,
SPLIT(line.customer, "'") AS customerID,
SPLIT(line.merchant, "'") AS merchantID,
SPLIT(line.age, "'") AS customerAge,
SPLIT(line.gender, "'") AS customerGender,
SPLIT(line.category, "'") AS transCategory

MERGE (customer:Customer {id: customerID[1], age: customerAge[1], gender: customerGender[1]})

MERGE (merchant:Merchant {id: merchantID[1]})

CREATE (transaction:Transaction {amount: line.amount, fraud: line.fraud, category: transCategory[1], step: line.step})-[:WITH]->(merchant)
CREATE (customer)-[:PERFORMS]->(transaction);


MATCH (c1:Customer)-[:PERFORMS]->(t1:Transaction)-[:WITH]->(m1:Merchant)
WITH c1, m1
MERGE (p1:LINK {id: m1.id})

MATCH (c1:Customer)-[:PERFORMS]->(t1:Transaction)-[:WITH]->(m1:Merchant)
WITH c1, m1, count(*) as cnt
MERGE (p2:LINK {id:c1.id})

MATCH (c1:Customer)-[:PERFORMS]->(t1:Transaction)-[:WITH]->(m1:Merchant)
WITH c1, m1, count(*) as cnt
MATCH (p1:LINK {id:m1.id})
WITH c1, m1, p1, cnt
MATCH (p2:LINK {id: c1.id})
WITH c1, m1, p1, p2, cnt
CREATE (p2)-[:PAYSTO {cnt: cnt}]->(p1)

MATCH (c1:Customer)-[:PERFORMS]->(t1:Transaction)-[:WITH]->(m1:Merchant)
WITH c1, m1, count(*) as cnt
MATCH (p1:LINK {id:c1.id})
WITH c1, m1, p1, cnt
MATCH (p2:LINK {id: m1.id})
WITH c1, m1, p1, p2, cnt
CREATE (p1)-[:PAYSTO {cnt: cnt}]->(p2)

// In memory graph creation
CALL gds.graph.project(
    'myGraph',
    'LINK',
    {
        PAYSTO: {
            type: 'PAYSTO',
            orientation: 'UNDIRECTED',
	 properties: {
                cnt: {
                    property: 'cnt',
                    defaultValue: 0.0 
                }
            }
        }
    }
)
YIELD graphName, nodeCount, relationshipCount;


// degree graph feature computation
CALL gds.degree.write('myGraph', {
    relationshipWeightProperty: 'cnt',
    writeProperty: 'degree'
})
YIELD nodePropertiesWritten;


// For viewing degree in database
MATCH (node:`LINK`)
WHERE exists(node.`degree`)
RETURN node, node.`degree` AS score
ORDER BY score DESC;


// PageRank graph feature computation
CALL gds.pageRank.write('myGraph', {
    writeProperty: 'pagerank'
})
YIELD nodePropertiesWritten, ranIterations;

CALL gds.pageRank.stream('myGraph', {
    maxIterations: 20,
    dampingFactor: 0.85
})
YIELD nodeId, score;


// For viewing PageRank in database
MATCH (node:LINK)
WHERE node.pagerank IS NOT NULL
RETURN node, node.pagerank AS score
ORDER BY score DESC;


// Betweenness graph feature computation
CALL gds.betweenness.write('myGraph', {
    writeProperty: 'betweenness'
})
YIELD nodePropertiesWritten;


// For viewing betweenness in database
MATCH (node:`LINK`)
WHERE exists(node.`betweenness`)
RETURN node, node.`betweenness` AS score
ORDER BY score DESC;


// Closeness graph features computation
CALL gds.closeness.write('myGraph', {
    writeProperty: 'closeness'
})
YIELD nodePropertiesWritten;


// For viewing in closeness database
MATCH (node:`LINK`)
WHERE exists(node.`closeness`)
RETURN node, node.`closeness` AS score
ORDER BY score DESC;


// louvain graph features computation
CALL gds.louvain.write('myGraph', {
    relationshipWeightProperty: null, 
    includeIntermediateCommunities: false,
    seedProperty: '',
    writeProperty: 'louvain'
})
YIELD communityCount, modularities, ranLevels;


// For viewing louvain in database
MATCH (node:LINK)
WHERE EXISTS(node.louvain)
WITH node, node.louvain AS community
WITH COLLECT(node) AS allNodes, community
RETURN community, allNodes AS nodes, SIZE(allNodes) AS size
ORDER BY size DESC;


// community graph features computation
CALL gds.labelPropagation.write('myGraph', {
    relationshipWeightProperty: 'cnt', 
    writeProperty: 'community'
})
YIELD communityCount, ranIterations, nodePropertiesWritten;


// For viewing louvain in database
MATCH (node:`LINK`)
WHERE exists(node.`community`)
WITH node.`community` AS community, collect(node) AS allNodes
RETURN community, allNodes AS nodes, size(allNodes) AS size
ORDER BY size DESC;

// For viewing degree, pagerank, community, betweenness in database
MATCH (p:LINK)
RETURN p.id AS id, p.pagerank as pagerank, p.degree as degree, p.community as community, p.betweenness as betweenness;


// connectedCommunity graph features computation
CALL gds.wcc.write('myGraph', {
    writeProperty: 'connectedCommunity'
})
YIELD componentCount, nodePropertiesWritten;


//For viewing connectedCommunity in database
MATCH (node:`LINK`)
WHERE exists(node.`connectedCommunity`)
WITH node.`connectedCommunity` AS community, collect(node) AS allNodes
RETURN community, allNodes AS nodes, size(allNodes) AS size
ORDER BY size DESC;


// Community graph features computation
CALL gds.triangleCount.write('myGraph', {
    writeProperty: 'trianglesCount'
})
YIELD nodeCount, globalTriangleCount;


// For viewing connectedCommunity in database
MATCH (node:`LINK`)
WHERE exists(node.`trianglesCount`)
RETURN node, node.`trianglesCount` AS triangles
ORDER BY triangles DESC;


// node similarity graph features computation
CALL gds.nodeSimilarity.write('myGraph', {
    similarityCutoff: 0.1,
    degreeCutoff: 1,
    writeProperty: 'similarity',
    writeRelationshipType: 'SIMILAR_JACCARD'
})
YIELD nodesCompared, relationshipsWritten;


// For viewing similarity in database
MATCH (from)-[rel:`SIMILAR_JACCARD`]-(to)
WHERE exists(rel.`similarity`)
RETURN from, to, rel.`similarity` AS similarity
ORDER BY similarity DESC;


// node cluster coefficient graph features computation
CALL gds.localClusteringCoefficient.write('myGraph', {
    writeProperty: 'coefficientCluster'
})
YIELD nodePropertiesWritten, averageClusteringCoefficient;


//For viewing cluster coefficient in database
MATCH (node:`LINK`)
WHERE exists(node.`coefficientCluster`)
RETURN node, node.`coefficientCluster` AS coefficient
ORDER BY coefficient DESC;
