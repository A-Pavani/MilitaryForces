<?php

include("./inc/had.php");
include("./include/function.php");

if(isset($_REQUEST['submit'])){
    include("./inc/db_conn.php");
    $unitname=$_REQUEST['unitname'];
    $unitid=$_REQUEST['unitid'];  
    $unittype=$_REQUEST['unittype'];
    $unitlocation=$_REQUEST['unitlocation'];
    $noofsoldiers=$_REQUEST['noofsoldiers'];
   
    $sql="INSERT INTO unit_table(unitname,unitid,unittype,unitlocation,noofsoldiers)
    VALUES('$unitname,','$unitid','$unittype','$unitlocation','$noofsoldiers')";
    if(mysqli_query($conn,$sql)){
        my_alert();
    }else{
        echo "Error :" .$sql."<br>".mysqli_error($conn);
    }
    mysqli_close($conn);
}
?>
<div class="icon">
        <h2 class="logo">Military Units</h2>
    </div> 
    <form  action="units.php" method="post">
    <!-- <fieldset> -->
    <div class="container">
    Enter Unitname : <input type="text" name="unitname" placeholder="Enter unit name" class="box-1"><br/>
    Enter unit ID : <input type="number" name="unitid" placeholder="Enter unit ID" class="box-2"><br/>
    Enter Unittype : <input type="text" name="unittype" placeholder="Enter unit type" class="box-3"><br/>
    Enter location : <input type="text"  name="unitlocation" placeholder="Enter base location" class="box-4"><br/>
    Enter number of soldiers :<input type="number" name="noofsoldiers" placeholder="Enter number of soldiers in the unit" class="box-5"><br/>
    <button class="btnn"  type="submit" name="submit" >Submit</button>
    <button class="btnn"><a href="combinedpage.php">Back</a></button>
    </div>
    <!-- </fieldset> -->
    </form>


<?php

include("./inc/fat.php");

?>
