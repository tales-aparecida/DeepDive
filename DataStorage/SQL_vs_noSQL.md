# Data storage models

## SQL vs NoSQL

|  SQL            | noSQL |
|  ---            |  ---  |
| Strong Schema   | Heterogeneous data |
| Tables          | Collections        |
| Rows            | Documents          |
| Join/Relations  | Duplication        |
| Split between tables |  |


### SQL or Relational Database Model System
Often refered as SQL Databases, relational databases were conceived in the 70's by E. F. Codd, based on mathematical set theory, and soon defined by the Structured Query Language (SQL). RDBMS 


### NoSQL

Reduce amount of inserts. 
Offers horizontal scalability (multiple data servers).
Loss of data consistency (not ACID).

## Relational tables, column family, document-orientad, graph

### 
#### MongoDB
The most popular NoSQL database, it's a document-oriented one
#### Amazon DynamoDB
A cloud based NoSQL database, that supports both key-value and document models and also has a library for geospatial indexing.

### Graph databases

Besides relational and analytical databases there is another popular data storage: Graph databases.
Those are kind of intermediary, as they do not have schemas but they are based around relationships between objects, the diffenrence, thoughm is that instead of joining normalized tables the graph db stores the relationship inside the object, which allows for quick queries. 

### Popular Databases
#### Neo4j
#### Microsoft Azure Cosmos DB
#### JanusGraph
#### TigerGraph
Parallelization to reach beyond two hops


### Languages 
#### Cypher
#### Gremlin
#### SPARQL


## Hybrid / Multi-model
### F1
F1, by Google, is a hybrid database that combines high availability, the scalability of NoSQL systems like Bigtable, and the consistency and usability of traditional SQL databases. It also includes a fully functional distributed SQL query engine and automatic change tracking and publishing.

### ArangoDB
Shown as a great option performance-wise by their benchmark tests, ArangoDB is a multi-model database, working well when compared to Postgres and MongoDB.

## References 
https://youtu.be/ZS_kXvOeQ5Y
https://youtu.be/GekQqFZm7mA
https://www.infoworld.com/article/3263764/database/what-is-a-graph-database-a-better-way-to-store-connected-data.html
https://www.infoworld.com/article/3269604/nosql/tigergraph-the-parallel-graph-database-explained.html
https://www.infoworld.com/article/3260184/nosql/how-to-choose-the-right-nosql-database.html
https://ai.google/research/pubs/pub41344
https://www.arangodb.com/2018/02/nosql-performance-benchmark-2018-mongodb-postgresql-orientdb-neo4j-arangodb/
https://blog.couchbase.com/comparison-sql-nosql-simplify-database-decision/
https://www.researchgate.net/publication/261079289_A_performance_comparison_of_SQL_and_NoSQL_databases
https://history-computer.com/ModernComputer/Software/Codd.html
