<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Table with Database</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 50px;
            opacity: 0;
            animation: fadeIn 2s forwards;
        }
        th, td {
            padding: 12px;
            border: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            animation: slideIn 1s ease-out forwards;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr {
            animation: fadeInRows 0.5s ease-in-out forwards;
        }
        h1, h2 {
            text-align: center;
            margin: 20px 0;
        }
        h1 {
            font-size: 60px;
            color: orange;
            margin-top: 0;
            animation: textSlideIn 1s ease-out forwards;
        }
        h2 {
            font-size: 40px;
            color: #333;
        }
        p {
            text-align: center;
            margin: 20px 0;
            color: #888;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideIn {
            from { transform: translateY(-20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        @keyframes fadeInRows {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes textSlideIn {
            from { transform: translateX(-100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
    </style>
</head>
<body>
    <h1>DETAILS OF MILITARY FORCES</h1>
    <p>_________________________________________________________________________________________________________________________________________________________</p>
    
    <h2> Soldier details</h2>
    <table>
        <tr>
           <th>First Name</th>
           <th>Last Name</th> 
           <th>Unit ID</th>
           <th>Contact Number</th>
           <th>Date of Birth</th>
           <th>Rank</th>
           <th>Email</th>
           <th>Gender</th>
        </tr>
        <?php
            $conn = mysqli_connect("localhost", "root", "", "soldier_details");
            if ($conn->connect_error) {
                die("Connection failed: " . $conn->connect_error);
            }
            $sql = "SELECT firstname, lastname, unitid, phonenumber, dateofbirth, rank, email, gender FROM soldier_table";
            $result = $conn->query($sql);

            if ($result->num_rows > 0) {
                while ($row = $result->fetch_assoc()) {
                    echo "<tr><td>" . $row["firstname"] . "</td><td>" . $row["lastname"] . "</td><td>" . $row["unitid"] . "</td><td>" . $row["phonenumber"] . "</td><td>" . $row["dateofbirth"] . "</td><td>" . $row["rank"] . "</td><td>" . $row["email"] . "</td><td>" . $row["gender"] . "</td></tr>";
                }
            } else {
                echo "<tr><td colspan='8'>No results found</td></tr>";
            }
            $conn->close();
        ?>
    </table>

    <h2> Equipment details</h2>
    <table>
        <tr>
           <th>Missile Name</th>
           <th>Missile ID</th>
           <th>Missile type</th>
           <th>Missile status</th>
        </tr>
        <?php
            $conn = mysqli_connect("localhost", "root", "", "equipment_details");
            if ($conn->connect_error) {
                die("Connection failed: " . $conn->connect_error);
            }
            $sql = "SELECT missilename,missileid,missiletype,missilestatus FROM equipment_table";
            $result = $conn->query($sql);

            if ($result->num_rows > 0) {
                while ($row = $result->fetch_assoc()) {
                    echo "<tr><td>" . $row["missilename"] . "</td><td>" . $row["missileid"] . "</td><td>" . $row["missiletype"] . "</td><td>" . $row["missilestatus"] . "</td></tr>";
                }
            } else {
                echo "<tr><td colspan='8'>No results found</td></tr>";
            }
            $conn->close();
        ?>
    </table>

    <h2> Unit details</h2>
    <table>
        <tr>
           <th>Unit Name</th>
           <th>Unit ID</th>
           <th>Unit type</th>
           <th>Unit location</th>
           <th>No of soldiers</th>
        </tr>
        <?php
            $conn = mysqli_connect("localhost", "root", "", "unit_details");
            if ($conn->connect_error) {
                die("Connection failed: " . $conn->connect_error);
            }
            $sql = "SELECT unitname,unitid,unittype,unitlocation,noofsoldiers FROM unit_table";
            $result = $conn->query($sql);

            if ($result->num_rows > 0) {
                while ($row = $result->fetch_assoc()) {
                    echo "<tr><td>" . $row["unitname"] . "</td><td>" . $row["unitid"] . "</td><td>" . $row["unittype"] . "</td><td>" . $row["unitlocation"] . "</td><td>". $row["noofsoldiers"] . "</td></tr>";
                }
            } else {
                echo "<tr><td colspan='8'>No results found</td></tr>";
            }
            $conn->close();
        ?>
    </table>

    <h2> Operation details</h2>
    <table>
        <tr>
           <th>Operation Name</th>
           <th>Operation ID</th>
           <th>Operation type</th>
           <th>Starting date</th>
           <th>Ending date</th>
        </tr>
        <?php
            $conn = mysqli_connect("localhost", "root", "", "unit_details");
            if ($conn->connect_error) {
                die("Connection failed: " . $conn->connect_error);
            }
            $sql = "SELECT operationname,operationid,operationtype,startingdate,endingdate FROM operations_table";
            $result = $conn->query($sql);

            if ($result->num_rows > 0) {
                while ($row = $result->fetch_assoc()) {
                    echo "<tr><td>" . $row["operationname"] . "</td><td>" . $row["operationid"] . "</td><td>" . $row["operationtype"] . "</td><td>" . $row["startingdate"] . "</td><td>". $row["endingdate"] . "</td></tr>";
                }
            } else {
                echo "<tr><td colspan='8'>No results found</td></tr>";
            }
            $conn->close();
        ?>
    </table>
</body>
</html>
