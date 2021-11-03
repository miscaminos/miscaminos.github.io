---
layout: post           									# (require) default post layout
title: "Big Data Analytics I"            # (require) a string title
date: 2021-11-03       									# (require) a post date
categories: [BigDataDatabase]   # (custom) some categories, but makesure these categories already exists inside path of `category/`
tags: [BigDataAnalytics]           	# (custom) tags only for meta `property="article:tag"`
---



Big Data Analytics

# NoSQL

### what is NoSQL

NoSQL = Not only SQL

family of databases that vary widely in style and tech

sharing common traits: non-relational (not a "column&row" type of data)

since 2000, NoSQL became important DB as scalable technology was needed to accommodate larger spectrum of for audience and public and as big data became important for business & society.

#### characteristics of NoSQL databases

4 common characteristics that describe NoSQL databases :

- key-value

- document

- column based

- graph style

What do NoSQL databases have in common? their roots are in open source community. Some companies developed commercial version and continue to support the open source community version. (e.g., IBM cloudant, Datastax, mongoDB) These different NoSQL databases differ technically, but they have few commonalities. 

Most NoSQL databases:

- are built to scale honrizontally
- share data more easily than relational database management system (RDBMS)
- use a global unique key to simplify data sharding (logical and physical ways of breaking up the database table into chunks)
- more use case specific than RDBMS
- more developer friendly than RDBMS
- allow more agile development via flexible schemas

Benefits of NoSQL (why use NoSQL?)

1. scalability

   server - server cluster - server racks - data centers

2. performance

   fast response, high concurrency

3. availability

   more resilient solution than single solution

4. cloud architecture

5. cost

6. flexible schema

   can build application specific database easily

7. varies data structures

8. specialized capabilities

   specific indexing, querying, modern HTTP APIs, Data replication robustness

   

### NoSQL Database Category

#### 1. Key-value NoSQL database 

**pro:** 

all data is stored as key & associated value blob

has least complex architecture and is represented as hashmap, so is ideal for basic CRUD operations

scale well and can be sharded easily

**con:**

not intended for complex query

atomic for single key operations only

value blobs are opaque to database (즉, less flexible data indexing and querying) 

#### suitable use cases

quick basic CRUD operation on non-interconnected data (e.g., storing and retrieving session information for web applications)

all transactions are based on the unique key (complex queries are not much needed)

storing in-app user profiles and preference within an application(e.g., shopping cart data for online stores)

#### unsuitable use cases

when data is interconnected (many - to - many relationship) (e.g., social networks, recommendation engine scenarios)

requiring high level consistency for multi -operational transaction with multiple keys (for such case, a database that handles ACID transactions is required) ACID = Atomicity, Consistency, Isolation, and Durability

when apps runs queries based on value vs. key, 'document' category NoSQL is more suitable

#### key-value NoSQL examples: 

Amazon DynamoDB, Oracle NoSQL, Redis, Aerospike, Riak KV, MemcacheDB, Project Voldemort

#### 2. Document

uses documents to make values visible and able to be queried

each piece of data is considered a document (typically in JSON or XML format)

each document offers a flexible schema (no two documents need to contain the same information)

content of document databases can be indexed and queried (e.g., ability to index and query the contents of the documents, offer key and value range lookups and search ability, perhaps analytical queries through paradigms like MapReduce)

document databases are horizontally scalable and allow for sharding across multiple nodes(sharded by some unique key in the document)

document stores also typically only guarantee atomic transactions on single document operations

#### suitable use cases

-event logging for apps and processes - each event instance is represented by a new document

-online blogs - each user, post, comment, like, or action is represented by a document

-operational datasets and metadata for web and mobile apps - designed with internet in mind (JSON, RESTful APIs, unstructured data)

#### unsuitable use cases

-when ACID transactions are required (즉, document database can't handle transactions that operate over multiple documents. Then relational database would be a better choice)

-if data is in an aggregate-oriented design(data naturally falls into a normalized tabular model. Then relational database would be a better choice )

#### document based NoSQL examples:

IBM Cloudant, Apache CouchDB, MongoDB, Terrastore, OrientDB, RavenDB

#### 3. Column

spawned from Google's Bigtable storage system

so, column based databases are a.k.a "Columnar databases", "Wide-column databases"

store and access data in columns or groups of columns

Column 'Families' are several rows, each with a unique key or identifier that belong to one or more columns. 

These columns are grouped together in families because they are often accessed together.

Rows in a column 'Family' are not required to share any of the same columns. They can share all, a subset, or none of the columns. And columns can be added to any number of rows and not to others.

#### suitable use cases

great for large amounts of sparse data. and can handle being deployed across clusters of nodes

-event logging and blogs(data stored in different fashion. for enterprise logging, every application can write to its own set of columns and have each row key formatted in such a way to promote easy lookup based on application and timestamp)

-"Counters" - a unique use case for Column-based database. can be used as an easy way to count or increment as events occur. (e.g., Cassandra have special column types that allow for simple counters)

-columns can have a time-to-live parameter - making them useful for data with an expiration date or time(e.g., trial periods or ad timing)

#### unsuitable use cases

When traditional ACID transactions (provided by relational databases) is required. (Reads and writes are only atomic at the row level.)

In early development, query patterns may change and require numerous changes to the column-based designs. (This can be costly and slow down the production timeline.)

#### column based NoSQL examples:

Cassandra, HBASE, Hypertable, and accumulo

#### 4. Graph 

stands apart from previous three categories. 좀 특별함.

(from a high level)graph databases store information in entities(or nodes) and relationships(or edges)

good to store data set that resembles a graph-like data structure

Sharding a graph database is not recommended since traversing a graph with nodes split across multiple servers can become difficult and hurt performance

graph databases are ACID transaction compliant (unlike other NoSQL databases) This attribute prevents any dangling relationships between nodes that don't exist.

#### suitable use cases:

can be very powerful when your data is highly connected and related in some way. (e.g., social networking, routing, spatial and map apps)

useful for applications that finds close locations or building shortest routes for directions.

recommendation engines that can leverage the close relationships and links between products to easily provide other options to customers

#### unsuitable use cases:

when looking for some advantages offered by other NoSQL database categories. 

when application needs to scale horizontally, you'll face limitations.

when trying to update all or a subset of nodes with a given parameter- operation will be very difficult

#### graph based NoSQL examples:

neo4j, OrientDB, ArangoDB, Amazon web services, Apache Giraph, JanusGraph



### Database deployment options

when choosing the best database deployment option for your application, you need to ask yourself:

"where to host your database?"

"how to manage your database?"

the goal is to find the simplest and most cost-effective option



- In-house-do-it-yourself scenario: 

  own setup of the underlying hardware and operating system, installation and configuration of chosen database management system.

  overall administration including patching and support and how application's data is designed

- hosted database solution:

  provider chooses and provides which hardware and operating system your database runs on. 

  your responsibility to provide and install the software and perform the administrative tasks, and design application's data

- Fully managed database-as-a-service (DBaaS)

  meant to eliminate the complexity and risk of doing it all in-house

  help developers to get to market faster

  assist development team to focus on developing the application functionalities

  


### choosing an appropriate data layer

important things to consider:

1. database functionality and performance - types of queries you need to ask your database

   - how long will you wait for answers (in a mobile and web application is interactive response needed.?) -> NoSQL database

   - if application requires data warehousing for batch analytics -> relational database(Hadoop-based technology)

   - how big your database is going to be/ if you don't know definite size of DB/ need scalable solution (database that will grow as applications grows)

2. database scalability - need to be scaled horizontally? if the application is running in the cloud, need database solution to be compatible with underlying architecture. Many NoSQL databases offer horizontal scalability that fits will with cloud architectures

3. database durability -increased risk of losing data when server crashes, so data durability is paramount  

4. consistency and transactional requirements (RDBMS provide strong consistency and transactional rollback capabilities) NoSQL databases operate inherently in a cluster and therefore can meet high availability requirements. 

5. data replication - important feature to achieve disaster recovery purposes . store data in additional data centers and allow for syncing to application clients for offline access. 

6. database geography

7. schema flexibility - flexible schema may be needed for rapid development where your data may change over time. some suitable NoSQL don't require downtime while making schema changes

8. database integration - whether your database layer can integrate easily with your application layer ( for web and mobile applications that uses JSON, use NoSQL database that also uses JSON)

9. database admin and developer resources



### ACID vs BASE

![acid_base](C:\SJL\스터디\acid_base.PNG)

*ACID: Atomicity, Consistency, Isolation, Durability <-RDBMS*

atomic: all operations in a transaction either 성공 or 완전 rollback(즉, every operation is rolled back if the transaction is not successfully completed)

consistent: on the completion of a transaction, the structural integrity of the data in the database is NOT compromised

isolated: transactions cannot compromise the integrity of other transactions by interacting with them while they are still in progress

durable: data related to the completed transaction will persist even in the case of network or power outages. (if a transaction fails, it will not impact the already changed data)

use cases examples: financial institutions almost always require ACID databases - tasks like handling money transfer requires atomic nature of ACID transactions

*BASE: Basic Availability, soft-state, Eventual consistence <-NoSQL*

basic availability: rather than enforcing immediate consistency, BASE-modelled NoSQL database ensure availability of data by spreading & replicating it across the nodes of the database cluster

soft-state: due to lack of immediate consistency, data values may change over time

eventually consistent: immediate consistency가 만들어지지 않는다고해서 consistency가 없는것은 아니다. eventually consistency가 이루어지지만, until ti does data reads might be inconsistent

use cases examples: marketing and customer service companies that deal with sentiment analysis and social network research. social media applications that contain huge amount of data and need to be available at all times.

### Distributed Databases

distributed database is a collection of multiple interconnected databases

it is spread physically across various locations which communicate via computer network

distributed database is physically distributed across data sites by fragmenting and replicating the data

follows the BASE consistency model to store a large piece of data on all servers of a distributed system 

To break the large data into smaller pieces to it can be distributed, fragmentation (a.k.a partitioning or sharding) of data is done by some NoSQL databases



In order to prevent distributed pieces of data from being eliminated/erased and getting lost, data at each node is relicated and store redundntly 

all data fragmentations are stored redundantly in two or more sites

disadvantage - replicated data need to be synchronized. 



### the CAP theorem (Brewer's theorem)

CAP theorem can be used to classify NoSQL databases. 

#### consistency vs. availability

consistency : whether a system operates fully or not. (does all nodes within a cluster see all the data they are supposed to?)

availability: simply means availability. (does each request get a response outside of failure or success?)

#### partition tolerance

partition: a communications break within a distributed system(a lost of temporarily delayed connection between nodes)

partition tolerance: cluster must continue to work despite any number of communication breakdowns between nodes in the system. 

(partition tolerance = the system continues to operate despite data loss or network failures)

In distributed systems, partition can't be avoided, so partition tolerance is a basic feature of native distributed system such as NoSQL

CAP theorem can be used to classify NoSQL databases. NoSQL databases choose between availability and consistency. and partition tolerance is a basic feature of NoSQL databases. Each NoSQL database has more weight on particular attribute(e.g., Apache Cassandra's primary functionalities are more towards availability, but doesn't mean Cassandra serves availability only. It means Apache Cassandra's primary attribute for the design and functionalities are closer to availability than consistencu)

![CAP_theorem](C:\SJL\스터디\CAP_theorem.PNG)



### Challenges in migrating from RDBMS to NoSQL Databases

migration from RDBMS to NoSQL could be triggered by requirements of performance driven by data volume, or flexibility in the schema or system scalability. 

#### RDBMS or NOSQL

NoSQL cannot be a replacement of RDBMS since RDBMS and NoSQL cater to different use cases. But a solution CAN use both RDBMS and NoSQL in order to fulfill the need of the application. 

RDBMS: mainly for consistency, structured da(fixed schenma), transaction, joins

NoSQL: mainly for high performance, unstructured data(flexible schema), availability, easy scalability

In situations where RDBMS is needed, solution design starts from the data, entities and their relationship. In contrast, NoSQL considers the way your application accesses the data and queries you are going to make.

In NoSQL, models should be based on how the application interacts with the data, rather than how the model can be stored as rows in tables. 

In RDBMS, data is normalized while, in NoSQL, data is denormalized. (denormalization of database is done to enhance read performance of the database)

In NoSQL, you start with your queries, not your data. Starting queries means you structure your data on disk accordingly. Think how data can be structured based on your queries. So you need to store the same data in different models just to answer the question. This leads to data denormalization.

when migrating from relational DBMS to NoSQL databases, you need to be aware that sometimes service requires availability more than consistency. 

there are many cases (esp. online services) which value availability more than consistency. so factors like considering the amount of data they are dealing with, and their geographical presence are becoming more important when choosing a database system. 

NoSQL is not designed to support transactions, joins or complex processing.

open source NoSQL Databases: MongoDB, Apache Cassandra



## MongoDB

what is MongoDB

- MongoDB is a document and NoSQL database. 
- uses MQL as query language (based on JavaScript)

Handling MongoDB/ use cases

- MongoDB shell is an interactive command line tool provided by MongoDB to interact with your databases
- CRUD operations consist of Create, Read, Update, and Delete 
- Indexes help quickly locate data. MongoDB stores data being indexed on the index entry and a location of the document on disk. MongoDB stores data being indexed on the index entry and a location of the document on disk.
- using an aggregation framework, you can perform complex analysis on the data in MongoDB. You can build your aggregation process in stages such as match, group, project, and sort
- replication is the duplication of data and any changes made to the data. Replication provides fault tolerance, redundancy, and high availability for your data.
- for growing data sets, you can use sharding to scale horizontally. MongoClient is a class that helps you interact with MongoDB.

### common use cases

- mainly used for search related use cases (where input data can be represented as key:document type of entries)

- It's easy to access by indexing
- it supports various data types including dates and numbers
- database schema can be flexible when working with MongoDB
- can change the database schema as needed without involving complex data definition language statements
- complex data analysis cna be done on the server using Aggregation Piepelines
- scalability makes it easier to work across the globe
- enables you to perform real-time analysis on your data



## Apache Cassandra

Apache Cassandra is an open-source, distributed, decentralized, elastically scalable, highly available, fault-tolerant, tunable and consistent database.

Its distribution is based on Amazon's Dynamo and its data model on Google's Bigtable. It's created at Facebook. 

uses CQL as query language (primary language to communicate with Apache Cassandra clusters)

This database is used by Netflix, Spotify and Uber

Cassnadra is suitable for cases when you need to record data extremely rapidly and make it available immediately for read operations (while hundreds of thousands of requests are generated) Recording transaction from an online shop or storing user access info/profile for services like Netflix.

Data consistency에 중점을 두는 MongoDB와는 다르게 Cassandra는 fast storage of the data, easy retrieval of data by key, available at all times, fast scalability, and geographical distribution of the servers의 강점을 가지고있다.

MongoDB는 Primary-Secondary architecture를 가지고있지만, Cassandra는 더 간단한 peer- to-peer architecture을 가지고있다. 

Cassandra의 주요 특징:

- favors availability over consistency
- available, scalable, fault tolerant due to distributed & decentralized architecture
- stores data in tables (tables are grouped in keyspaces)
- extremely fast write throughput while maintaining cluster performance for other operations (read capability to scale clusters is extremely fast in a linear fashion - without the need to restart or recognize
- NOT a drop-in replacement of relational database. since Cassandra lacks thee major characteristics of relational database as following:
  - does not support join
  - has limited aggregations to support
  - has limited support for transactions

- write operation이 atomic, isolated, and durable in nature, but does not guarantee consistency (hence not suitable for banking applications)

#### Common use cases:

write-intensive (number of writes exceeds number of reads) - Cassandra provides high write throughput by having no read before write dafult.

e.g., storing all the clicks on your website or all the access attempts on your service

when application doesn't require that many updates or deletes

when data access done via known primary key("partition key") and key allows even spread of data inside the cluster 

when theres no need to joins or complex aggregations as part of your queries

Cassandra is good for "always available" type of applications (Netflix, Uber, Spotify, etc...)



## IBM Cloudant

IBM's DBaaS built on Apache CouchDB uses JSON document store

distributed database optimized for large workloads, web, mobile, IoT and serverless app

offers a powerful replication protocol

this cloud architecture provides high availability, disaster recovery, and optimal performance

주요 장점:

scalability, availability, durability, partition tolerance and online upgrades, also has improved security and querying by using a document database
