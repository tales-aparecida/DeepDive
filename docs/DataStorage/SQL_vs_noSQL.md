# Data storage models

One important part of analysing data is where and how to store it. Many variables should be considered to make a choice, and each use case might require a different kind of storage, but knowledge is a great one to start, given that knowing full well how to use your tools can take you beyond the freshest _buzzwordy_ ones, influencing in other variables like efficiency, coding speed and storage footprint, for example.

Here will be listed some of the popular database paradygms, along with their well known implementions on the market. Let's start talking about the most widespread model for the last 40 years.

## SQL or Relational Database Model System (RDBMS)
Often refered as SQL Databases, relational databases were conceived in the 70's by _E. F. Codd_, based on mathematical set theory, soon assisted by Structured Query Language (SQL), which, as the name implies, is a language definition to create querys to interact with data in four ways, with what could be called sub-languages: data query language (DQL), a data definition language (DDL), a data control language (DCL), and a data manipulation language (DML). As it has been around for almost as long as RDBMS, it ended up naming the category.

RDBMS are modeled as tables, made from homogeneous rows, that is, rows with the same strongly-typed columns. Each table usually represents a real world entity or concept, so, in order to retrieve some useful information from the database we must cross data between tables, and that is achieved by the **Join** clause.

Mentioning this clause first was completely intentional, even if a little non-didatic, because that is one of the main differences of RDBMS and non-relational DBs, the relationship.

Let's model a simple example based on the \*[W3 schools SQL course](https://www.w3schools.com/sql). Say we have a trading store where we register who bought what from whom, so, a simple RDBMS model would have three main tables:

- Customer(CustomerID, CustomerName)
- Product(ProductID, Price, ProductName)
- Employee(EmployeeID, EmployeeName)

Pretty straight forward, right? Customer and Seller have two columns whilst Item has three. In order to actually store a trade, though, we need another table that links our entities:

- Order(CustomerID, ProductID, EmployeeID)

So, our resulting query would look something like this:

```sql
SELECT EmployeeName, CustomerName, ProductName, Price
FROM (((Orders
INNER JOIN Customer ON Orders.CustomerID = Customer.CustomerID)
INNER JOIN Employee ON Orders.EmployeeID = Employee.EmployeeID)
INNER JOIN Product  ON Orders.ProductID  = Product.ProductID)
```

We could create indexes on the IDs and optimize this query, but in the end we would still need to reach four tables and cross their data. One may argue that we could save all to just one table, but that would be an antipattern, as RDBMS focus on ACID.

// Paragrafo explicando ACID e LOCKS 


To summarize, RDMBS have:

- Strong typed and static schema  
- Tables (rows and columns)  
- Relationships between entities connected through Joins
- Indexes  
- Triggers, Constraints and Views  


## NoSQL or non-relation databases


Reduce amount of inserts. 
Offers horizontal scalability (multiple data servers) given the lack of joins (data duplication).
Loss of data consistency, as it is not ACID (Atomicity, Consistency, Isolation, and Durability), but instead CAP (Consistency, Availability, Partition tolerance) or BASE (Basically Available, Soft state, Eventual consistency).

### Key-Value pairs

### Wide-column
#### Cassandra
- Lacks a LIKE operator
- Has one table per query pattern, which introduces lots of redundancy in order to avoid joins

### Document-oriented

#### MongoDB
The most popular NoSQL database, and the fourth overall is a document-oriented one. It has great writing and reading times, as some benchmarks show, but it can't always give consistency on models that have explicit relationships, which would have data redundancy.

MongoDB has a limited left outer join as of 3.2, while most of RDBMS implementations have at least four types of join operations: inner join, left outer join, right outer join and full outer join.

##### Shard key

#### Amazon DynamoDB
A cloud based NoSQL database, that supports both key-value and document models and also has a library for geospatial indexing.


### Graph databases

Besides relational and analytical databases there is another popular data storage: Graph databases.
Those are kind of intermediary, as they do not have schemas but they are based around relationships between objects, the diffenrence, thoughm is that instead of joining normalized tables the graph db stores the relationship inside the object, which allows for quick queries. 

#### Neo4j
#### Microsoft Azure Cosmos DB
#### JanusGraph
#### TigerGraph
Parallelization to reach beyond two hops


## Hybrid / Multi-model
### F1
F1, by Google, is a hybrid database that combines high availability, the scalability of NoSQL systems like Bigtable, and the consistency and usability of traditional SQL databases. It also includes a fully functional distributed SQL query engine and automatic change tracking and publishing.

### ArangoDB
Shown as a great option performance-wise by their benchmark tests, ArangoDB is a multi-model database, working well when compared to Postgres and MongoDB.

## Outline

|  SQL (relational)           | noSQL |
|  ---            |  ---  |
| Strong Schema   | Heterogeneous data |
| Tables          | Collections        |
| Rows            | Documents          |
| Join/Relations  | Duplication        |
| Split between tables |  |

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
https://www.researchgate.net/publication/314639479_Comparison_of_Relational_Document_and_Graph_Databases_in_the_Context_of_the_Web_Application_Development
https://en.wikipedia.org/wiki/Unstructured_data
http://www.sarahmei.com/blog/2013/11/11/why-you-should-never-use-mongodb/
https://hackernoon.com/mongodb-indexes-and-performance-2e8f94b23c0a

*
The actual query on w3schoolds would be:

```sql
SELECT FirstName + " " + LastName AS EmployeeName, CustomerName, ProductName, Price
FROM ((((Orders
INNER JOIN OrderDetails ON Orders.OrderID = OrderDetails.OrderID)
INNER JOIN Customers ON Orders.CustomerID = Customers.CustomerID)
INNER JOIN Employees ON Orders.EmployeeID = Employees.EmployeeID)
INNER JOIN Products ON OrderDetails.ProductID = Products.ProductID);
```

### Languages 
#### Cypher
#### Gremlin
#### SPARQL

