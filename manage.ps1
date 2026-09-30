param(
    [Parameter(Position = 0)]
    [string]$Command,

    [Parameter(Position = 1)]
    [string]$Profile,

    [Parameter(Position = 2)]
    [string]$Destination,

    [switch]$Delete,

    [switch]$NoDelete,

    [switch]$Open
)

$ErrorActionPreference = "Stop"

$CommandHelp = [ordered]@{
    "build-database" = [ordered]@{
        Description = "Adds source data to the Neo4j database."
        Usage = @(
            ".\manage.ps1 build-database <profile> -Delete",
            ".\manage.ps1 build-database <profile> -NoDelete"
        )
        Options = @(
            "-Delete      Delete the existing Neo4j database volume and create a new database.",
            "-NoDelete    Keep the existing Neo4j database and add the new data to it."
        )
    }

    "generate-dataset" = [ordered]@{
        Description = "Resets the processed embeddings and generates the dataset using the existing Neo4j database."
        Usage = @(
            ".\manage.ps1 generate-dataset <profile>"
        )
        Options = @()
    }

    "build-and-generate" = [ordered]@{
        Description = "Adds source data to the Neo4j database and then generates the dataset."
        Usage = @(
            ".\manage.ps1 build-and-generate <profile> -Delete",
            ".\manage.ps1 build-and-generate <profile> -NoDelete"
        )
        Options = @(
            "-Delete      Reset the Neo4j database before adding the source data.",
            "-NoDelete    Keep the existing Neo4j database and add the new data to it."
        )
    }

    "upload-data" = [ordered]@{
        Description = "Uploads a local data folder to a selected location on the Oracle VM."
        Usage = @(
            ".\manage.ps1 upload-data <source> <destination>"
        )
        Options = @(
            "<source>     A folder relative to the local data directory.",
            "<destination> A path relative to VM_DATA_PATH on the Oracle VM."
        )
    }

    "add-new-profile" = [ordered]@{
        Description = "Creates a new profile from the template in template\profile."
        Usage = @(
            ".\manage.ps1 add-new-profile <profile>",
            ".\manage.ps1 add-new-profile <profile> -Open"
        )
        Options = @(
            "-Open        Opens emotional-anchor.json and profile.json in Visual Studio Code after creation."
        )
    }

    "open-profile" = [ordered]@{
        Description = "Opens emotional-anchor.json and profile.json for a profile in Visual Studio Code."
        Usage = @(
            ".\manage.ps1 open-profile <profile>"
        )
        Options = @()
    }

    "list-profiles" = [ordered]@{
        Description = "Lists all available profiles."
        Usage = @(
            ".\manage.ps1 list-profiles"
        )
        Options = @()
    }

    "delete-profile" = [ordered]@{
        Description = "Deletes an existing profile after confirmation."
        Usage = @(
            ".\manage.ps1 delete-profile <profile>"
        )
        Options = @()
    }
}

function Show-Help {
    param(
        [string]$Topic
    )

    if ($null -eq $Topic) {
        $Topic = ""
    }
    else {
        $Topic = $Topic.ToLower()
    }

    if ([string]::IsNullOrWhiteSpace($Topic)) {
        Write-Host ""
        Write-Host "Project Epithet management commands:" -ForegroundColor Cyan
        Write-Host ""

        foreach ($entry in $CommandHelp.GetEnumerator()) {
            Write-Host ("  {0,-20} {1}" -f $entry.Key, $entry.Value.Description)
        }

        Write-Host ""
        Write-Host "Get detailed help with:" -ForegroundColor Cyan
        Write-Host "  .\manage.ps1 help <command>"
        Write-Host ""

        return
    }

    if ($Topic -notin $CommandHelp.Keys) {
        Write-Host ""
        Write-Host "Unknown help topic '$Topic'." -ForegroundColor Red
        Write-Host ""
        Write-Host "Use '.\manage.ps1 help' to see available commands." -ForegroundColor Cyan
        Write-Host ""

        return
    }

    $help = $CommandHelp[$Topic]

    Write-Host ""
    Write-Host $Topic -ForegroundColor Cyan
    Write-Host $help.Description
    Write-Host ""

    Write-Host "Usage:" -ForegroundColor Cyan

    foreach ($usage in $help.Usage) {
        Write-Host "  $usage"
    }

    if ($help.Options.Count -gt 0) {
        Write-Host ""
        Write-Host "Options:" -ForegroundColor Cyan

        foreach ($option in $help.Options) {
            Write-Host "  $option"
        }
    }

    Write-Host ""
    Write-Host "Build and generation commands run detached in the background." -ForegroundColor DarkGray
    Write-Host ""
}

function Fail {
    param(
        [string]$Message,
        [string]$HelpTopic
    )

    Write-Host ""
    Write-Host "Error: $Message" -ForegroundColor Red

    if (-not [string]::IsNullOrWhiteSpace($HelpTopic)) {
        Write-Host "Use '.\manage.ps1 help $HelpTopic' for help." -ForegroundColor Cyan
    }
    else {
        Write-Host "Use '.\manage.ps1 help' to see available commands." -ForegroundColor Cyan
    }

    Write-Host ""
    exit 1
}

function Get-DotEnvValue {
    param(
        [string]$Name
    )

    $envPath = Join-Path $PSScriptRoot ".env"

    if (-not (Test-Path $envPath -PathType Leaf)) {
        throw "Could not find .env at '$envPath'."
    }

    foreach ($line in Get-Content $envPath) {
        $line = $line.Trim()

        if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith("#")) {
            continue
        }

        if ($line -match "^\s*$([regex]::Escape($Name))\s*=\s*(.*)\s*$") {
            $value = $Matches[1].Trim()

            if (
                ($value.StartsWith('"') -and $value.EndsWith('"')) -or
                ($value.StartsWith("'") -and $value.EndsWith("'"))
            ) {
                $value = $value.Substring(1, $value.Length - 2)
            }

            return $value
        }
    }

    throw "Could not find '$Name' in '$envPath'."
}

function Confirm-Operation {
    param(
        [string]$Message
    )

    Write-Host ""
    Write-Host $Message -ForegroundColor Yellow
    Write-Host "Profile: $Profile" -ForegroundColor Cyan
    Write-Host ""

    $confirmation = Read-Host "Are you sure you want to continue? (y/N)"

    if ($confirmation -notmatch '^(y|yes)$') {
        Write-Host ""
        Write-Host "Operation cancelled." -ForegroundColor Cyan
        exit 0
    }
}

function Reset-Embeddings {
    param(
        [string]$Name
    )

    $processedPath = Join-Path $PSScriptRoot "data\profiles\$Name\processed"

    if (-not (Test-Path $processedPath -PathType Container)) {
        return
    }

    Write-Host ""
    Write-Host "Resetting processed embeddings..." -ForegroundColor Yellow

    Get-ChildItem -Path $processedPath -Force |
        Remove-Item -Recurse -Force

    Write-Host "Processed embeddings reset successfully." -ForegroundColor Green
}

function Remove-Neo4jData {
    Write-Host ""
    Write-Host "Removing Neo4j container and volume..." -ForegroundColor Yellow

    docker compose down -v neo4j

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to remove Neo4j container and volume."
    }
}

function Start-Detached {
    param(
        [string]$Service
    )

    Write-Host ""
    Write-Host "Starting '$Service' in the background..." -ForegroundColor Cyan

    $containerId = docker compose run --rm --build --detach `
        -e "PROFILE=$Profile" `
        $Service

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start '$Service'."
    }

    $containerId = $containerId.Trim()

    Write-Host ""
    Write-Host "Started successfully." -ForegroundColor Green
    Write-Host "Container ID: $containerId" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Follow the output with:" -ForegroundColor Cyan
    Write-Host "  docker logs -f $containerId"
    Write-Host ""
}

function Upload-DataFolder {
    param(
        [string]$Source,
        [string]$Destination
    )

    if ([string]::IsNullOrWhiteSpace($Source)) {
        Fail "No source folder was provided." "upload-data"
    }

    if ([string]::IsNullOrWhiteSpace($Destination)) {
        Fail "No destination path was provided." "upload-data"
    }

    $localPath = Join-Path $PSScriptRoot "data\$Source"

    if (-not (Test-Path $localPath -PathType Container)) {
        Fail "Local data folder '$localPath' does not exist." "upload-data"
    }

    $vmHost = Get-DotEnvValue "VM_HOST"
    $vmDataPath = Get-DotEnvValue "VM_DATA_PATH"

    $Destination = $Destination.TrimStart("/", "\")
    $remotePath = "$vmDataPath/$Destination"

    Write-Host ""
    Write-Host "Checking whether '$remotePath' already exists on the VM..." -ForegroundColor Cyan

    ssh $vmHost "test ! -e '$remotePath'"

    if ($LASTEXITCODE -ne 0) {
        throw "Remote data path '$remotePath' already exists."
    }

    Write-Host ""
    Write-Host "This will upload '$localPath' to '$remotePath' on the Oracle VM." -ForegroundColor Yellow
    Write-Host ""

    $confirmation = Read-Host "Are you sure you want to continue? (y/N)"

    if ($confirmation -notmatch '^(y|yes)$') {
        Write-Host ""
        Write-Host "Operation cancelled." -ForegroundColor Cyan
        exit 0
    }

    Write-Host ""
    Write-Host "Uploading data folder..." -ForegroundColor Cyan

    scp -r $localPath "${vmHost}:$remotePath"

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upload '$localPath'."
    }

    Write-Host ""
    Write-Host "Data folder uploaded successfully." -ForegroundColor Green
}

function Open-ProfileConfig {
    param(
        [string]$Name
    )

    $profilePath = Join-Path $PSScriptRoot "data\profiles\$Name"

    if (-not (Test-Path $profilePath -PathType Container)) {
        Fail "Profile '$Name' does not exist." "open-profile"
    }

    $emotionalAnchorPath = Join-Path $profilePath "config\emotional-anchor.json"
    $profileConfigPath = Join-Path $profilePath "config\profile.json"

    $filesToOpen = @()

    if (Test-Path $emotionalAnchorPath -PathType Leaf) {
        $filesToOpen += $emotionalAnchorPath
    }
    else {
        Write-Host ""
        Write-Host "Warning: emotional-anchor.json was not found." -ForegroundColor Yellow
    }

    if (Test-Path $profileConfigPath -PathType Leaf) {
        $filesToOpen += $profileConfigPath
    }
    else {
        Write-Host ""
        Write-Host "Warning: profile.json was not found." -ForegroundColor Yellow
    }

    if ($filesToOpen.Count -eq 0) {
        throw "No profile configuration files were found."
    }

    if (-not (Get-Command code -ErrorAction SilentlyContinue)) {
        throw "The 'code' command was not found. Make sure Visual Studio Code is installed and its command-line interface is available."
    }

    Write-Host ""
    Write-Host "Opening profile configuration in Visual Studio Code..." -ForegroundColor Cyan

    code --reuse-window $filesToOpen
}

function Add-NewProfile {
    param(
        [string]$Name
    )

    $templatePath = Join-Path $PSScriptRoot "template\profile"
    $profilesPath = Join-Path $PSScriptRoot "data\profiles"
    $newProfilePath = Join-Path $profilesPath $Name

    if (-not (Test-Path $templatePath -PathType Container)) {
        Fail "Profile template '$templatePath' does not exist." "add-new-profile"
    }

    if (Test-Path $newProfilePath) {
        Fail "A profile named '$Name' already exists." "add-new-profile"
    }

    $invalidCharacters = [IO.Path]::GetInvalidFileNameChars()

    if ($Name.IndexOfAny($invalidCharacters) -ge 0) {
        Fail "Profile name '$Name' contains invalid characters." "add-new-profile"
    }

    if ($Name -eq "." -or $Name -eq "..") {
        Fail "Invalid profile name '$Name'." "add-new-profile"
    }

    Confirm-Operation "This will create the new profile '$Name' from the template."

    Write-Host ""
    Write-Host "Creating profile '$Name'..." -ForegroundColor Cyan

    New-Item -ItemType Directory -Path $newProfilePath -Force | Out-Null

    Get-ChildItem -Path $templatePath -Force |
        Copy-Item -Destination $newProfilePath -Recurse -Force

    Write-Host ""
    Write-Host "Profile '$Name' created successfully." -ForegroundColor Green

    if ($Open) {
        Open-ProfileConfig $Name
    }
}

function List-Profiles {
    $profilesPath = Join-Path $PSScriptRoot "data\profiles"

    if (-not (Test-Path $profilesPath -PathType Container)) {
        Fail "Profiles directory '$profilesPath' does not exist." "list-profiles"
    }

    $profiles = @(Get-ChildItem -Path $profilesPath -Directory | Sort-Object Name)

    Write-Host ""
    Write-Host "Available profiles:" -ForegroundColor Cyan
    Write-Host ""

    if ($profiles.Count -eq 0) {
        Write-Host "  No profiles found." -ForegroundColor DarkGray
    }
    else {
        foreach ($profile in $profiles) {
            Write-Host "  $($profile.Name)"
        }
    }

    Write-Host ""
}

function Delete-Profile {
    param(
        [string]$Name
    )

    $profilesPath = Join-Path $PSScriptRoot "data\profiles"
    $profilePath = Join-Path $profilesPath $Name

    if (-not (Test-Path $profilePath -PathType Container)) {
        Fail "Profile '$Name' does not exist." "delete-profile"
    }

    Confirm-Operation "This will permanently DELETE the profile '$Name'."

    Write-Host ""
    Write-Host "Deleting profile '$Name'..." -ForegroundColor Yellow

    Remove-Item -Path $profilePath -Recurse -Force

    Write-Host ""
    Write-Host "Profile '$Name' deleted successfully." -ForegroundColor Green
    Write-Host ""
}

if ([string]::IsNullOrWhiteSpace($Command)) {
    Fail "No command was provided."
}

$Command = $Command.ToLower()

if ($Command -eq "help") {

    if ($Delete -or $NoDelete -or $Open) {
        Fail "The help command does not accept flags."
    }

    if (-not [string]::IsNullOrWhiteSpace($Destination)) {
        Fail "The help command does not accept a third argument."
    }

    Show-Help $Profile
    exit 0
}

if ($Command -notin $CommandHelp.Keys) {
    Fail "Unknown command '$Command'."
}

if ($Command -eq "list-profiles") {

    if ($Delete -or $NoDelete -or $Open) {
        Fail "list-profiles does not accept flags." "list-profiles"
    }

    if (-not [string]::IsNullOrWhiteSpace($Profile)) {
        Fail "list-profiles does not accept a profile name." "list-profiles"
    }

    if (-not [string]::IsNullOrWhiteSpace($Destination)) {
        Fail "list-profiles does not accept a destination path." "list-profiles"
    }

    List-Profiles
    exit 0
}

if ($Command -eq "upload-data") {

    if ([string]::IsNullOrWhiteSpace($Profile)) {
        Fail "No source folder was provided." "upload-data"
    }

    if ([string]::IsNullOrWhiteSpace($Destination)) {
        Fail "No destination path was provided." "upload-data"
    }

}
else {

    if (-not [string]::IsNullOrWhiteSpace($Destination)) {
        Fail "$Command does not accept a destination path." $Command
    }

    if ([string]::IsNullOrWhiteSpace($Profile)) {

        if ($Command -eq "add-new-profile") {
            Fail "No profile name was provided." $Command
        }
        elseif ($Command -eq "open-profile") {
            Fail "No profile name was provided." $Command
        }
        elseif ($Command -eq "delete-profile") {
            Fail "No profile name was provided." $Command
        }
        elseif ($Command -eq "upload-data") {
            Fail "No source folder was provided." $Command
        }
        else {
            Fail "No profile was provided." $Command
        }
    }
}

$hasDelete = $Delete.IsPresent
$hasNoDelete = $NoDelete.IsPresent
$hasOpen = $Open.IsPresent

if (
    $Command -eq "generate-dataset" -or
    $Command -eq "upload-data" -or
    $Command -eq "open-profile" -or
    $Command -eq "delete-profile"
) {

    if ($hasDelete -or $hasNoDelete) {
        Fail "$Command does not accept -Delete or -NoDelete." $Command
    }

    if ($hasOpen) {
        Fail "$Command does not accept -Open." $Command
    }

}
elseif ($Command -eq "add-new-profile") {

    if ($hasDelete -or $hasNoDelete) {
        Fail "add-new-profile does not accept -Delete or -NoDelete." $Command
    }

}
else {

    if ($hasDelete -and $hasNoDelete) {
        Fail "Specify either -Delete or -NoDelete, not both." $Command
    }

    if (-not $hasDelete -and -not $hasNoDelete) {
        Fail "You must explicitly specify either -Delete or -NoDelete." $Command
    }

    if ($hasOpen) {
        Fail "$Command does not accept -Open." $Command
    }
}

switch ($Command) {

    "build-database" {

        if ($Delete) {
            Confirm-Operation "This will DELETE the existing Neo4j database volume and create a new database."
            Remove-Neo4jData
        }
        else {
            Confirm-Operation "This will KEEP the existing Neo4j database and add the new data to it."
        }

        Start-Detached "build-database"
    }

    "generate-dataset" {

        Confirm-Operation "This will reset the profile's processed embeddings and generate the dataset using the existing Neo4j database."
        Reset-Embeddings $Profile

        Start-Detached "generate-dataset"
    }

    "build-and-generate" {

        if ($Delete) {
            Confirm-Operation "This will reset the profile's processed embeddings, DELETE the existing Neo4j database volume, create a new database, and then generate the dataset."
            Reset-Embeddings $Profile
            Remove-Neo4jData
        }
        else {
            Confirm-Operation "This will reset the profile's processed embeddings, KEEP the existing Neo4j database, add the new data to it, and then generate the dataset."
            Reset-Embeddings $Profile
        }

        Start-Detached "build-and-generate"
    }

    "upload-data" {

        Upload-DataFolder $Profile $Destination
    }

    "add-new-profile" {

        Add-NewProfile $Profile
    }

    "open-profile" {

        Open-ProfileConfig $Profile
    }

    "list-profiles" {

        List-Profiles
    }

    "delete-profile" {

        Delete-Profile $Profile
    }
}