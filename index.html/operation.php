<?php

include("./inc/had.php");
include("./include/function.php");

if(isset($_REQUEST['submit'])){
    include("./inc/db_conn.php");
    $operationname=$_REQUEST['operationname'];
    $operationid=$_REQUEST['operationid'];  
    $operationtype=$_REQUEST['operationtype'];
    $startingdate=$_REQUEST['startingdate'];
    $endingdate=$_REQUEST['endingdate'];
   
    $sql="INSERT INTO operations_table(operationname,operationid,operationtype,startingdate,endingdate)
    VALUES('$operationname,','$operationid','$operationtype','$startingdate','$endingdate')";
    if(mysqli_query($conn,$sql)){
        my_alert();
    }else{
        echo "Error :" .$sql."<br>".mysqli_error($conn);
    }
    mysqli_close($conn);
}
?>

<div class="icon">
        <h2 class="logo">Military Operations</h2>
    </div> 
    <form  action="operation.php" method="POST">
    <!-- <fieldset> -->
    <div class="container">
    Enter operation name : <input type="text" name="operationname" placeholder="Enter operation name" class="box-1"><br/>
    Enter operation ID : <input type="number" name="operationid" placeholder="Enter operation ID" class="box-2"><br/>
    Enter operation type : <input type="text" name="operationtype" placeholder="Enter operation type" class="box-3"><br/>
    Enter starting date : <input type="date" size="30" name="startingdate" class="box-4"><br/>
    Enter ending date  : <input type="date" size="30" name="endingdate"class="box-5"><br/>
    <button class="btnn"  type="submit" name="submit" >Submit</button>
    <button class="btnn"><a href="combinedpage.php">Back</a></button>
    </div>
    <!-- </fieldset> -->
<?php

include("./inc/fat.php");

?>
